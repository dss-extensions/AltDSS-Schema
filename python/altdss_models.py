from __future__ import annotations
from typing import List, Optional, Union, TextIO, Dict
from typing_extensions import Annotated
from enum import Enum, IntEnum
from collections import defaultdict
import os.path
from io import StringIO
import json
import numpy as np

from pydantic import (
    field_validator,
    model_validator,
    AliasChoices,
    BaseModel,
    Field,
    FilePath,
    PositiveFloat,
    PositiveInt,
    RootModel,
    ValidationInfo,
)
from pydantic_complex import Complex

ARRAY_LIKE = (list, tuple, np.ndarray)

def _dump_symmetric_matrix(prop_name, value):
    if (value is None) or (len(value) == 0):
        return f' {prop_name}=()'

    last_col = len(value) - 1
    s = f' {prop_name}=('
    for i, col in enumerate(value):
        for element in col:
            s += f'{element} '

        if i != last_col:
            s += '|'

    s += ')'
    return s

def _check_symmetric_matrix(value_wrapped, order):
    if value_wrapped is None or value_wrapped.root is None:
        return value_wrapped

    value = value_wrapped.root

    # Check order
    if (order is None or order == 0) and len(value):
        raise ValueError("Expected None or an empty list.")

    if order != len(value):
        raise ValueError(f"Matrix order ({len(value)}) does not match the expected value ({order}).")

    max_cols = max(len(row) for row in value)
    if max_cols != order:
        raise ValueError(f"Matrix order ({len(value)}) does not match the expected value ({order}).")

    min_cols = min(len(row) for row in value)
    if min_cols == 0:
        raise ValueError(f"Symmetric matrix is not fully specified.")

    num_elements = sum(len(row) for row in value)
    if num_elements != (order * order) and num_elements != ((order * (order + 1)) // 2):
        raise ValueError(f"Provide either a triangular matrix or a full matrix.")

    # If there is redundant data (e.g. when a full matrix is provided instead of a tringular matrix),
    # check if the redundant elements match
    if min_cols == max_cols:
        # TODO: this should be slow, try to use NumPy or something better?
        for i in range(order):
            col_i = tuple(row[i] for row in value)
            row_i = tuple(value[i])
            if col_i != row_i:
                raise ValueError('Data provided for symmetric matrix is not symmetric.')

        return value_wrapped

    # TODO: validate other variations?
    return value_wrapped


def _quoted(s: Union[str, Enum]):
    if isinstance(s, Enum):
        s = s.value
    else:
        s = str(s)

    for ch in ' ()':
        if ch in s:
            if '"' not in s:
                return f'"{s}"'
            elif "'" not in s:
                return f"'{s}'"
            else:
                return f"[{s}]"

    return s


def _quoted_list(lst):
    return ', '.join(_quoted(s) for s in lst)


def _complex_to_list(c):
    if isinstance(c, complex):
        return [c.real, c.imag]

    assert len(c) == 2
    return c


def _csvfile_array_length(fn, ncols=None):
    # We need to read the whole file here to get the number of points
    with open(fn, 'rb') as fcsv:
        contents = fcsv.read()
        nrows = contents.count(b'\n')

    # Check if last line is empty, e.g.
    # ----
    # 1\n
    # 2\n
    # 3\n
    #
    # ----
    # vs
    # ----
    # 1\n
    # 2\n
    # 3\n
    # 4
    # ----

    if contents[-1] != '\n':
        nrows += 1

    return nrows

_pqcsvfile_array_length = _csvfile_array_length

def _dblfile_array_length(fn, ncols=2):
    fsize = os.path.getsize(fn)
    if (fsize > 0) and (fsize % (8 * ncols) != 0):
        raise ValueError(f'Invalid file size: "{fn}"')

    return fsize // (8 * ncols)


def _sngfile_array_length(fn, ncols=2):
    fsize = os.path.getsize(fn)
    if (fsize > 0) and (fsize % (4 * ncols) != 0):
        raise ValueError(f'Invalid file size: "{fn}"')

    return fsize // (4 * ncols)


def _as_list(arr):
    if isinstance(arr, np.ndarray):
        return list(arr.ravel())

    return arr


def _filepath_array(val):
    fn = val.get('CSVFile')
    if fn is not None:
        column = val.get('Column', 1)
        header = val.get('Header', False)

        num = _csvfile_array_length(fn)

        if header:
            num -= 1

        return (num, f'(File={_quoted(fn)} Column={column} Header={header})')

    fn = val.get('DblFile')
    if fn is not None:
        num = _dblfile_array_length(fn, ncols=1)
        return (num, f'(DblFile={_quoted(fn)})')

    fn = val.get('SngFile')
    if fn is not None:
        num = _sngfile_array_length(fn, ncols=1)
        return (num, f'(SngFile={_quoted(fn)})')

    raise ValueError('Unexpected value for array from file') #TODO: better message


def _filepath_stringlist(val, length=False):
    if isinstance(val, (tuple, list)): #TODO: Sequence?
        if not length:
            return _quoted_list(val)

        return len(val), _quoted_list(val)

    fn = val.get('File')
    if fn is not None:
        if not length:
            return f'File={_quoted(fn)}'
        else:
            return _csvfile_array_length(fn), f'File={_quoted(fn)}'

    raise ValueError('Unexpected value for list of strings') #TODO: better message


def _dump_dss_container(data, item_cls, output):
    is_vsource = (item_cls == Vsource)
    if isinstance(data, (list, tuple)):
        it = iter(data)
        if is_vsource:
            item = next(it)
            item_cls.dict_dump_dss(item, output, True)

        for item in it:
            item_cls.dict_dump_dss(item, output, False)

        return

    if not isinstance(data, dict):
        raise ValueError('Expected a dict')

    _JSONFile = data.get('JSONFile')
    if _JSONFile is not None:
        with open(_JSONFile, 'r') as f:
            data = json.load(f)

        _dump_dss_container(data, item_cls, output)
        return

    _JSONLinesFile = data.get('JSONLinesFile')
    if _JSONLinesFile is not None:
        with open(_JSONLinesFile, 'r') as f:
            it = iter(f)
            if is_vsource:
                item = json.loads(next(it))
                item_cls.dict_dump_dss(item, output, True)

            for line in it:
                item = json.loads(line)
                item_cls.dict_dump_dss(item, output, False)

        return


class DynInitType(RootModel):
    root: Dict[str, Union[float, str]]

    def dumps_dss(self, edit=False) -> str:
        """
        Dump the contents in the DSS script format to a string.

        Convenience function.
        """
        with StringIO() as output:
            self.dump_dss(output, edit)
            return output.getvalue()

    def dump_dss(self, output: TextIO, edit=False):
        """
        Dump the contents in the DSS script format to an `output` stream.
        """
        DynInitType.dict_dump_dss(self.model_dump(exclude_unset=True), output, edit=edit)

    @staticmethod
    def dict_dumps_dss(fields: Dict, edit=False) -> str:
        """
        Dump the contents in the DSS script format to a string.

        Convenience function.
        """
        with StringIO() as output:
            DynInitType.dict_dump_dss(fields, output, edit)
            return output.getvalue()

    @staticmethod
    def dict_dump_dss(fields: Dict, output: TextIO, edit=False):
        """
        Dump the contents in the DSS script format to an `output` stream.
        """
        for k, v in fields.items():
            output.write(f''' {k}={_quoted(v)}''')


class VisualizeQuantity(Enum):
    Currents = "Currents"
    Voltages = "Voltages"
    Powers = "Powers"


class ReductionStrategy(Enum):
    Default = "Default"
    ShortLines = "ShortLines"
    MergeParallel = "MergeParallel"
    BreakLoop = "BreakLoop"
    Dangling = "Dangling"
    Switches = "Switches"
    Laterals = "Laterals"


class EarthModel(Enum):
    Carson = "Carson"
    FullCarson = "FullCarson"
    Deri = "Deri"


EarthModel_ = EarthModel


class LineType(Enum):
    oh = "oh"
    ug = "ug"
    ug_ts = "ug_ts"
    ug_cn = "ug_cn"
    swt_ldbrk = "swt_ldbrk"
    swt_fuse = "swt_fuse"
    swt_sect = "swt_sect"
    swt_rec = "swt_rec"
    swt_disc = "swt_disc"
    swt_brk = "swt_brk"
    swt_elbow = "swt_elbow"
    busbar = "busbar"


LineType_ = LineType


class LengthUnit(Enum):
    none = "none"
    mi = "mi"
    kft = "kft"
    km = "km"
    m = "m"
    ft = "ft"
    in_ = "in"
    cm = "cm"
    mm = "mm"


class ScanType(Enum):
    None_ = "None"
    Zero = "Zero"
    Positive = "Positive"


ScanType_ = ScanType


class SequenceType(Enum):
    Negative = "Negative"
    Zero = "Zero"
    Positive = "Positive"


class Connection(Enum):
    Wye = "wye"
    Delta = "delta"


class CoreType(Enum):
    Shell = "shell"
    OnePhase = "1-phase"
    ThreeLeg = "3-leg"
    FourLeg = "4-leg"
    FiveLeg = "5-leg"
    CoreOnePhase = "core-1-phase"


class PhaseSequence(Enum):
    Lag = "Lag"
    Lead = "Lead"


class LoadSolutionModel(Enum):
    PowerFlow = "PowerFlow"
    Admittance = "Admittance"


class RandomType(Enum):
    None_ = "None"
    Gaussian = "Gaussian"
    Uniform = "Uniform"
    LogNormal = "LogNormal"


class ControlMode(Enum):
    Off = "Off"
    Static = "Static"
    Event = "Event"
    Time = "Time"
    MultiRate = "MultiRate"


ControlMode_ = ControlMode


class InverterControlMode(Enum):
    GFL = "GFL"
    GFM = "GFM"


class SolutionMode(Enum):
    Snapshot = "Snap"
    Daily = "Daily"
    Yearly = "Yearly"
    M1 = "M1"
    LD1 = "LD1"
    PeakDay = "PeakDay"
    DutyCycle = "DutyCycle"
    Direct = "Direct"
    MF = "MF"
    FaultStudy = "FaultStudy"
    M2 = "M2"
    M3 = "M3"
    LD2 = "LD2"
    AutoAdd = "AutoAdd"
    Dynamic = "Dynamic"
    Harmonic = "Harmonic"
    Time = "Time"
    HarmonicT = "HarmonicT"


class SolutionAlgorithm(Enum):
    Normal = "Normal"
    Newton = "Newton"


class CircuitModel(Enum):
    Multiphase = "Multiphase"
    Positive = "Positive"


class AutoAddDeviceType(Enum):
    Generator = "Generator"
    Capacitor = "Capacitor"


class LoadShapeClass(Enum):
    None_ = "None"
    Daily = "Daily"
    Yearly = "Yearly"
    Duty = "Duty"


class MonitoredPhaseEnum(Enum):
    min = "min"
    max = "max"
    avg = "avg"


class MonitoredPhase(RootModel[Union[MonitoredPhaseEnum, Annotated[int, Field(ge=1)]]]):
    root: Union[MonitoredPhaseEnum, Annotated[int, Field(ge=1)]] = Field(..., title="Monitored Phase")


class PlotProfilePhasesEnum(Enum):
    Default = "Default"
    All = "All"
    Primary = "Primary"
    LL3Ph = "LL3Ph"
    LLAll = "LLAll"
    LLPrimary = "LLPrimary"


class PlotProfilePhases(RootModel[Union[PlotProfilePhasesEnum, Annotated[int, Field(ge=1)]]]):
    root: Union[PlotProfilePhasesEnum, Annotated[int, Field(ge=1)]] = Field(..., title="Plot: Profile Phases")


class LoadShapeAction(Enum):
    Normalize = "Normalize"
    DblSave = "DblSave"
    SngSave = "SngSave"


class LoadShapeInterpolation(Enum):
    Avg = "Avg"
    Edge = "Edge"


class TShapeAction(Enum):
    DblSave = "DblSave"
    SngSave = "SngSave"


class PriceShapeAction(Enum):
    DblSave = "DblSave"
    SngSave = "SngSave"


class VSourceModel(Enum):
    Thevenin = "Thevenin"
    Ideal = "Ideal"


class LoadModel(IntEnum):
    ConstantPQ = 1
    ConstantZ = 2
    Motor = 3
    CVR = 4
    ConstantI = 5
    ConstantP_FixedQ = 6
    ConstantP_FixedX = 7
    ZIPV = 8


class LoadStatus(Enum):
    Variable = "Variable"
    Fixed = "Fixed"
    Exempt = "Exempt"


class RegControlPhaseSelectionEnum(Enum):
    min = "min"
    max = "max"


class RegControlPhaseSelection(RootModel[Union[RegControlPhaseSelectionEnum, Annotated[int, Field(ge=1)]]]):
    root: Union[RegControlPhaseSelectionEnum, Annotated[int, Field(ge=1)]] = Field(..., title="RegControl: Phase Selection")


class CapControlType(Enum):
    Current = "Current"
    Voltage = "Voltage"
    kvar = "kvar"
    Time = "Time"
    PowerFactor = "PowerFactor"
    Follow = "Follow"


class DynamicExpDomain(Enum):
    Time = "Time"
    dq = "dq"


class GeneratorModel(IntEnum):
    ConstantPQ = 1
    ConstantZ = 2
    ConstantPV = 3
    ConstantP_FixedQ = 4
    ConstantP_FixedX = 5
    UserModel = 6
    ApproxInverter = 7


class GeneratorDispatchMode(Enum):
    Default = "Default"
    LoadLevel = "LoadLevel"
    Price = "Price"


class GeneratorStatus(Enum):
    Variable = "Variable"
    Fixed = "Fixed"


class StorageState(Enum):
    Charging = "Charging"
    Idling = "Idling"
    Discharging = "Discharging"


class StorageDispatchMode(Enum):
    Default = "Default"
    LoadLevel = "LoadLevel"
    Price = "Price"
    External = "External"
    Follow = "Follow"


class StorageControllerDischargeMode(Enum):
    Peakshave = "Peakshave"
    Follow = "Follow"
    Support = "Support"
    Loadshape = "Loadshape"
    Time = "Time"
    Schedule = "Schedule"
    IPeakshave = "I-Peakshave"


class StorageControllerChargeMode(Enum):
    Loadshape = "Loadshape"
    Time = "Time"
    PeakshaveLow = "PeakshaveLow"
    IPeakshaveLow = "I-PeakshaveLow"


class RelayType(Enum):
    Current = "Current"
    Voltage = "Voltage"
    ReversePower = "ReversePower"
    F46 = "46"
    F47 = "47"
    Generic = "Generic"
    Distance = "Distance"
    TD21 = "TD21"
    DOC = "DOC"


class RelayState(Enum):
    closed = "closed"
    open = "open"
    trip = "trip"


class RecloserState(Enum):
    closed = "closed"
    open = "open"
    trip = "trip"


class FuseAction(Enum):
    close = "close"
    open = "open"


class FuseState(Enum):
    closed = "closed"
    open = "open"


class SwtControlState(Enum):
    closed = "closed"
    open = "open"


class PVSystemModel(IntEnum):
    ConstantP_PF = 1
    ConstantY = 2
    UserModel = 3


class UPFCMode(IntEnum):
    Off = 0
    VoltageRegulator = 1
    PhaseAngleRegulator = 2
    DualRegulator = 3
    DoubleReference_Voltage = 4
    DoubleReference_Dual = 5


class ESPVLControlType(Enum):
    SystemController = "SystemController"
    LocalController = "LocalController"


class IndMach012SlipOption(Enum):
    VariableSlip = "VariableSlip"
    FixedSlip = "FixedSlip"


class AutoTransConnection(Enum):
    Wye = "wye"
    Delta = "delta"
    Series = "series"


class InvControlControlMode(Enum):
    Voltvar = "Voltvar"
    VoltWatt = "VoltWatt"
    DynamicReaccurr = "DynamicReaccurr"
    WattPF = "WattPF"
    Wattvar = "Wattvar"
    AVR = "AVR"
    GFM = "GFM"


class InvControlCombiMode(Enum):
    VV_VW = "VV_VW"
    VV_DRC = "VV_DRC"


class InvControlVoltageCurveXRef(Enum):
    Rated = "Rated"
    Avg = "Avg"
    RAvg = "RAvg"


class InvControlVoltWattYAxis(Enum):
    PAvailablePU = "PAvailablePU"
    PMPPPU = "PMPPPU"
    PctPMPPPU = "PctPMPPPU"
    KVARatingPU = "KVARatingPU"


class InvControlRateOfChangeMode(Enum):
    Inactive = "Inactive"
    LPF = "LPF"
    RiseFall = "RiseFall"


class InvControlReactivePowerReference(Enum):
    VARAVAL = "VARAVAL"
    VARMAX = "VARMAX"


class InvControlControlModel(IntEnum):
    Linear = 0
    Exponential = 1


class GICTransformerType(Enum):
    GSU = "GSU"
    Auto = "Auto"
    YY = "YY"


class VSConverterControlMode(Enum):
    Fixed = "Fixed"
    PacVac = "PacVac"
    PacQac = "PacQac"
    VdcVac = "VdcVac"
    VdcQac = "VdcQac"


class MonitorAction(Enum):
    Clear = "Clear"
    Save = "Save"
    TakeSample = "TakeSample"
    Process = "Process"
    Reset = "Reset"


class EnergyMeterAction(Enum):
    Allocate = "Allocate"
    Clear = "Clear"
    Reduce = "Reduce"
    Save = "Save"
    TakeSample = "TakeSample"
    ZoneDump = "ZoneDump"


class SymmetricMatrix(RootModel[List[List[float]]]):
    root: List[List[float]]




class FloatArray(RootModel[List[float]]):
    root: List[float]

class FloatArrayFromCSV(BaseModel):
    CSVFile: FilePath = Field(..., title="CSVFile")
    Column: Optional[PositiveInt] = Field(None, title="Column")
    Header: Optional[bool] = Field(None, title="Header")

class FloatArrayFromDbl(BaseModel):
    DblFile: FilePath = Field(..., title="DblFile")

class FloatArrayFromSng(BaseModel):
    SngFile: FilePath = Field(..., title="SngFile")


class ArrayOrFilePath(RootModel[Union[FloatArray, FloatArrayFromCSV, FloatArrayFromDbl, FloatArrayFromSng]]):
    root: Union[FloatArray, FloatArrayFromCSV, FloatArrayFromDbl, FloatArrayFromSng] = Field(..., title="ArrayOrFilePath")


    @model_validator(mode="before")
    def _val_dss_model(self) -> "ArrayOrFilePath":
        try:
            _fields_set = set(self.keys())
        except:
            if isinstance(self, (list, tuple)):
                return self

            raise

        # Validate oneOf (spec. sets) based on the `required` lists
        _required_FloatArray = "root" in _fields_set and isinstance(self.root, FloatArray)
        _required_FloatArrayFromCSV = _fields_set.issuperset({'CSVFile'})
        _required_FloatArrayFromDbl = _fields_set.issuperset({'DblFile'})
        _required_FloatArrayFromSng = _fields_set.issuperset({'SngFile'})
        num_specs = _required_FloatArray + _required_FloatArrayFromCSV + _required_FloatArrayFromDbl + _required_FloatArrayFromSng
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self




class StringArray(RootModel[List[Annotated[str, Field(min_length=1)]]]):
    root: List[Annotated[str, Field(min_length=1)]]

class StringArrayFromFile(BaseModel):
    File: FilePath = Field(..., title="File")


class StringArrayOrFilePath(RootModel[Union[StringArray, StringArrayFromFile]]):
    root: Union[StringArray, StringArrayFromFile] = Field(..., title="StringArrayOrFilePath")


    @model_validator(mode="before")
    def _val_dss_model(self) -> "StringArrayOrFilePath":
        try:
            _fields_set = set(self.keys())
        except:
            if isinstance(self, (list, tuple)):
                return self

            raise

        # Validate oneOf (spec. sets) based on the `required` lists
        _required_StringArray = "root" in _fields_set and isinstance(self.root, StringArray)
        _required_StringArrayFromFile = _fields_set.issuperset({'File'})
        num_specs = _required_StringArray + _required_StringArrayFromFile
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



class JSONFilePath(BaseModel):
    JSONFile: FilePath = Field(..., title="JSONFile")



class JSONLinesFilePath(BaseModel):
    JSONLinesFile: FilePath = Field(..., title="JSONLinesFile")



class Bus(BaseModel):
    Name: Annotated[str, Field(max_length=255, min_length=1)] = Field(..., title="Name")
    X: Optional[float] = Field(None, title="X")
    Y: Optional[float] = Field(None, title="Y")
    kVLN: Optional[PositiveFloat] = Field(None, title="kVLN")
    kVLL: Optional[PositiveFloat] = Field(None, title="kVLL")
    Keep: Optional[bool] = Field(None, title="Keep")

    def dumps_dss(self, edit=False) -> str:
        """
        Dump the DSS object dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            self.dump_dss(output, edit)
            return output.getvalue()

    def dump_dss(self, output: TextIO, edit=False):
        """
        Dump the DSS object to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        Bus.dict_dump_dss(self.model_dump(exclude_unset=True), output, edit=edit)

    @staticmethod
    def dict_dumps_dss(fields: Dict, edit=False) -> str:
        """
        Dump the DSS object from a dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            Bus.dict_dump_dss(fields, output, edit)
            return output.getvalue()

    @staticmethod
    def dict_dump_dss(fields: Dict, output: TextIO, edit=False):
        """
        Dump the DSS object from a dict to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        
        if len(fields) <= 1:
            return # skip, nothing useful to do
        
        _Name = _quoted(fields['Name'])
        _X, _Y, _kVLN, _Keep = fields.get('X'), fields.get('Y'), fields.get('kVLN'), fields.get('Keep')
        if _X is not None:
            output.write(f'SetBusXY {_Name} {_X} {_Y}\n')
        if _kVLN is not None:
            output.write(f'SetkVBase {_Name} kVLN={_kVLN}\n')
        else:
            _kVLL = fields.get('kVLL')
            if _kVLL is not None:
                output.write(f'SetkVBase {_Name} kVLL={_kVLL}\n')
        if _Keep:
            output.write(f'KeepList ({_Name} )\n')
        

    @model_validator(mode="before")
    def _val_dss_model(self) -> "Bus":
        _fields_set = set(self.keys())
        _anyOf_Bus_0 = _fields_set.issuperset({'kVLN'}) and (not {'kVLL'}.issubset(_fields_set))
        _anyOf_Bus_1 = _fields_set.issuperset({'kVLL'}) and (not {'kVLN'}.issubset(_fields_set))
        _anyOf_Bus_2 = _fields_set.issuperset({}) and (not {'kVLL', 'kVLN'}.issubset(_fields_set))
        anyOf_OK = any([_anyOf_Bus_0, _anyOf_Bus_1, _anyOf_Bus_2])
        if not anyOf_OK:
            raise ValueError("Conflict detected in the provided properties. Only one specification type is allowed.")

        return self



Bus_ = Bus


class BusConnection(RootModel[Annotated[str, Field(pattern='[^.]+(\\.[0-9]+)*')]]):
    root: Annotated[str, Field(pattern='[^.]+(\\.[0-9]+)*')]



class LineCode_Common(BaseModel):
    Name: Annotated[str, Field(max_length=255, min_length=1)] = Field(..., title="Name")
    Like: Optional[str] = Field(None, title="Like")
    NPhases: Optional[int] = Field(None, title="NPhases")
    Units: Optional[LengthUnit] = Field(None, title="Units")
    BaseFreq: Optional[PositiveFloat] = Field(None, title="BaseFreq")
    NormAmps: Optional[float] = Field(None, title="NormAmps")
    EmergAmps: Optional[float] = Field(None, title="EmergAmps")
    FaultRate: Optional[float] = Field(None, title="FaultRate")
    PctPerm: Optional[float] = Field(None, title="PctPerm")
    Repair: Optional[float] = Field(None, title="Repair")
    Rg: Optional[float] = Field(None, title="Rg")
    Xg: Optional[float] = Field(None, title="Xg")
    rho: Optional[float] = Field(None, title="rho")
    Neutral: Optional[int] = Field(None, title="Neutral")
    Ratings: Optional[ArrayOrFilePath] = Field(None, title="Ratings")
    LineType: Optional[LineType_] = Field(None, title="LineType")
    Kron: Optional[bool] = Field(None, title="Kron")

class LineCode_Z0Z1C0C1(LineCode_Common):
    R1: float = Field(..., title="R1")
    X1: float = Field(..., title="X1")
    R0: Optional[float] = Field(None, title="R0")
    X0: Optional[float] = Field(None, title="X0")
    C1: float = Field(..., title="C1")
    C0: Optional[float] = Field(None, title="C0")

class LineCode_ZMatrixCMatrix(LineCode_Common):
    RMatrix: SymmetricMatrix = Field(..., title="RMatrix")
    XMatrix: SymmetricMatrix = Field(..., title="XMatrix")
    CMatrix: Optional[SymmetricMatrix] = Field(None, title="CMatrix")

    @field_validator('RMatrix')
    @classmethod
    def _RMatrix_check_symm_matrix(cls, value: Optional[SymmetricMatrix], info: ValidationInfo) -> Optional[SymmetricMatrix]:
        return _check_symmetric_matrix(value, info.data.get('NPhases'))

    @field_validator('XMatrix')
    @classmethod
    def _XMatrix_check_symm_matrix(cls, value: Optional[SymmetricMatrix], info: ValidationInfo) -> Optional[SymmetricMatrix]:
        return _check_symmetric_matrix(value, info.data.get('NPhases'))

    @field_validator('CMatrix')
    @classmethod
    def _CMatrix_check_symm_matrix(cls, value: Optional[SymmetricMatrix], info: ValidationInfo) -> Optional[SymmetricMatrix]:
        return _check_symmetric_matrix(value, info.data.get('NPhases'))


class LineCode(RootModel[Union[LineCode_Z0Z1C0C1, LineCode_ZMatrixCMatrix]]):
    root: Union[LineCode_Z0Z1C0C1, LineCode_ZMatrixCMatrix] = Field(..., title="LineCode")

    def dumps_dss(self, edit=False) -> str:
        """
        Dump the DSS object dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            self.dump_dss(output, edit)
            return output.getvalue()

    def dump_dss(self, output: TextIO, edit=False):
        """
        Dump the DSS object to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        LineCode.dict_dump_dss(self.model_dump(exclude_unset=True), output, edit=edit)

    @staticmethod
    def dict_dumps_dss(fields: Dict, edit=False) -> str:
        """
        Dump the DSS object from a dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            LineCode.dict_dump_dss(fields, output, edit)
            return output.getvalue()

    @staticmethod
    def dict_dump_dss(fields: Dict, output: TextIO, edit=False):
        """
        Dump the DSS object from a dict to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        if edit:
            output.write(f'''edit LineCode.{fields['Name']}''')
        else:
            output.write(f'''new LineCode.{fields['Name']}''')

        _Like = fields.get('Like')
        if _Like is not None:
            output.write(f' Like={_quoted(_Like)}')

        _NPhases = fields.get('NPhases')
        if _NPhases is not None:
            output.write(f' NPhases={_NPhases}')

        _R1 = fields.get('R1')
        if _R1 is not None:
            output.write(f' R1={_R1}')

        _X1 = fields.get('X1')
        if _X1 is not None:
            output.write(f' X1={_X1}')

        _R0 = fields.get('R0')
        if _R0 is not None:
            output.write(f' R0={_R0}')

        _X0 = fields.get('X0')
        if _X0 is not None:
            output.write(f' X0={_X0}')

        _C1 = fields.get('C1')
        if _C1 is not None:
            output.write(f' C1={_C1}')

        _C0 = fields.get('C0')
        if _C0 is not None:
            output.write(f' C0={_C0}')

        _Units = fields.get('Units')
        if _Units is not None:
            output.write(f' Units={_quoted(_Units)}')

        _RMatrix = fields.get('RMatrix')
        if _RMatrix is not None:
            output.write(_dump_symmetric_matrix("RMatrix", _RMatrix))

        _XMatrix = fields.get('XMatrix')
        if _XMatrix is not None:
            output.write(_dump_symmetric_matrix("XMatrix", _XMatrix))

        _CMatrix = fields.get('CMatrix')
        if _CMatrix is not None:
            output.write(_dump_symmetric_matrix("CMatrix", _CMatrix))

        _BaseFreq = fields.get('BaseFreq')
        if _BaseFreq is not None:
            output.write(f' BaseFreq={_BaseFreq}')

        _NormAmps = fields.get('NormAmps')
        if _NormAmps is not None:
            output.write(f' NormAmps={_NormAmps}')

        _EmergAmps = fields.get('EmergAmps')
        if _EmergAmps is not None:
            output.write(f' EmergAmps={_EmergAmps}')

        _FaultRate = fields.get('FaultRate')
        if _FaultRate is not None:
            output.write(f' FaultRate={_FaultRate}')

        _PctPerm = fields.get('PctPerm')
        if _PctPerm is not None:
            output.write(f' PctPerm={_PctPerm}')

        _Repair = fields.get('Repair')
        if _Repair is not None:
            output.write(f' Repair={_Repair}')

        _Rg = fields.get('Rg')
        if _Rg is not None:
            output.write(f' Rg={_Rg}')

        _Xg = fields.get('Xg')
        if _Xg is not None:
            output.write(f' Xg={_Xg}')

        _rho = fields.get('rho')
        if _rho is not None:
            output.write(f' rho={_rho}')

        _Neutral = fields.get('Neutral')
        if _Neutral is not None:
            output.write(f' Neutral={_Neutral}')

        _Ratings = fields.get('Ratings')
        if _Ratings is not None:
            if isinstance(_Ratings, ARRAY_LIKE):
                output.write(f' Seasons={len(_Ratings)}')
                _Ratings = _as_list(_Ratings)
            else:
                _length_Ratings, _Ratings = _filepath_array(_Ratings)
                output.write(f' Seasons={_length_Ratings}')
            output.write(f' Ratings={_Ratings}')

        _LineType = fields.get('LineType')
        if _LineType is not None:
            output.write(f' LineType={_quoted(_LineType)}')

        _Kron = fields.get('Kron')
        if _Kron is not None:
            output.write(f' Kron={_Kron}')

        output.write('\n')

    @model_validator(mode="before")
    def _val_dss_model(self) -> "LineCode":
        _fields_set = set(self.keys())
        # Validate oneOf (spec. sets) based on the `required` lists
        _required_Z0Z1C0C1 = _fields_set.issuperset({'C1', 'R1', 'X1'})
        _required_ZMatrixCMatrix = _fields_set.issuperset({'RMatrix', 'XMatrix'})
        num_specs = _required_Z0Z1C0C1 + _required_ZMatrixCMatrix
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



LineCode_ = LineCode


class LineCodeList(RootModel[List[LineCode]]):
    root: List[LineCode]





class LineCodeContainer(RootModel[Union[LineCodeList, JSONFilePath, JSONLinesFilePath]]):
    root: Union[LineCodeList, JSONFilePath, JSONLinesFilePath] = Field(..., title="LineCodeContainer")


    @model_validator(mode="before")
    def _val_dss_model(self) -> "LineCodeContainer":
        try:
            _fields_set = set(self.keys())
        except:
            if isinstance(self, (list, tuple)):
                return self

            raise

        # Validate oneOf (spec. sets) based on the `required` lists
        _required_LineCodeList = "root" in _fields_set and isinstance(self.root, LineCodeList)
        _required_JSONFilePath = _fields_set.issuperset({'JSONFile'})
        _required_JSONLinesFilePath = _fields_set.issuperset({'JSONLinesFile'})
        num_specs = _required_LineCodeList + _required_JSONFilePath + _required_JSONLinesFilePath
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



class LoadShape_Common(BaseModel):
    Name: Annotated[str, Field(max_length=255, min_length=1)] = Field(..., title="Name")
    Like: Optional[str] = Field(None, title="Like")
    MemoryMapping: Optional[bool] = Field(None, title="MemoryMapping")
    Mean: Optional[float] = Field(None, title="Mean")
    StdDev: Optional[float] = Field(None, title="StdDev")
    UseActual: Optional[bool] = Field(None, title="UseActual")
    PMax: Optional[float] = Field(None, title="PMax")
    QMax: Optional[float] = Field(None, title="QMax")
    PBase: Optional[float] = Field(None, title="PBase")
    QBase: Optional[float] = Field(None, title="QBase")
    Interpolation: Optional[LoadShapeInterpolation] = Field(None, title="Interpolation")
    Action: Optional[LoadShapeAction] = Field(None, title="Action")

class LoadShape_PMultQMultHour(LoadShape_Common):
    Hour: ArrayOrFilePath = Field(..., title="Hour")
    QMult: Optional[ArrayOrFilePath] = Field(None, title="QMult")
    PMult: ArrayOrFilePath = Field(..., title="PMult")

class LoadShape_PMultQMultInterval(LoadShape_Common):
    Interval: Annotated[float, Field(ge=0)] = Field(..., title="Interval")
    QMult: Optional[ArrayOrFilePath] = Field(None, title="QMult")
    PMult: ArrayOrFilePath = Field(..., title="PMult")

class LoadShape_PQCSVFile(LoadShape_Common):
    PQCSVFile: FilePath = Field(..., title="PQCSVFile")

class LoadShape_CSVFile(LoadShape_Common):
    CSVFile: FilePath = Field(..., title="CSVFile")

class LoadShape_SngFile(LoadShape_Common):
    SngFile: FilePath = Field(..., title="SngFile")

class LoadShape_DblFile(LoadShape_Common):
    DblFile: FilePath = Field(..., title="DblFile")


class LoadShape(RootModel[Union[LoadShape_PMultQMultHour, LoadShape_PMultQMultInterval, LoadShape_PQCSVFile, LoadShape_CSVFile, LoadShape_SngFile, LoadShape_DblFile]]):
    root: Union[LoadShape_PMultQMultHour, LoadShape_PMultQMultInterval, LoadShape_PQCSVFile, LoadShape_CSVFile, LoadShape_SngFile, LoadShape_DblFile] = Field(..., title="LoadShape")

    def dumps_dss(self, edit=False) -> str:
        """
        Dump the DSS object dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            self.dump_dss(output, edit)
            return output.getvalue()

    def dump_dss(self, output: TextIO, edit=False):
        """
        Dump the DSS object to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        LoadShape.dict_dump_dss(self.model_dump(exclude_unset=True), output, edit=edit)

    @staticmethod
    def dict_dumps_dss(fields: Dict, edit=False) -> str:
        """
        Dump the DSS object from a dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            LoadShape.dict_dump_dss(fields, output, edit)
            return output.getvalue()

    @staticmethod
    def dict_dump_dss(fields: Dict, output: TextIO, edit=False):
        """
        Dump the DSS object from a dict to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        _length_NPts = None

        if edit:
            output.write(f'''edit LoadShape.{fields['Name']}''')
        else:
            output.write(f'''new LoadShape.{fields['Name']}''')

        _Like = fields.get('Like')
        if _Like is not None:
            output.write(f' Like={_quoted(_Like)}')

        _MemoryMapping = fields.get('MemoryMapping')
        if _MemoryMapping is not None:
            output.write(f' MemoryMapping={_MemoryMapping}')

        _Interval = fields.get('Interval')
        if _Interval is not None:
            output.write(f' Interval={_Interval}')

        _Hour = fields.get('Hour')
        if _Hour is not None:
            if isinstance(_Hour, ARRAY_LIKE):
                _length_NPts = len(_Hour)
                output.write(f' NPts={_length_NPts}')
                _Hour = _as_list(_Hour)
            else:
                _length_Hour, _Hour = _filepath_array(_Hour)
                if _length_NPts is None:
                    _length_NPts = _length_Hour
                    output.write(f' NPts={_length_NPts}')
                elif _length_NPts != _length_Hour:
                    raise ValueError(f'Array length ({_length_Hour}) for "Hour" (from file) does not match expected length ({_length_NPts})')

            output.write(f' Hour={_Hour}')

        _Mean = fields.get('Mean')
        if _Mean is not None:
            output.write(f' Mean={_Mean}')

        _StdDev = fields.get('StdDev')
        if _StdDev is not None:
            output.write(f' StdDev={_StdDev}')

        _CSVFile = fields.get('CSVFile')
        if _CSVFile is not None:
            _length_NPts = _csvfile_array_length(_CSVFile)
            output.write(f' NPts={_length_NPts}')
            output.write(f' CSVFile={_quoted(_CSVFile)}')

        _SngFile = fields.get('SngFile')
        if _SngFile is not None:
            _length_NPts = _sngfile_array_length(_SngFile, cols=2 if (fields.get('Interval') or 0.0) == 0.0 else 1)
            output.write(f' NPts={_length_NPts}')
            output.write(f' SngFile={_quoted(_SngFile)}')

        _DblFile = fields.get('DblFile')
        if _DblFile is not None:
            _length_NPts = _dblfile_array_length(_DblFile, cols=2 if (fields.get('Interval') or 0.0) == 0.0 else 1)
            output.write(f' NPts={_length_NPts}')
            output.write(f' DblFile={_quoted(_DblFile)}')

        _QMult = fields.get('QMult')
        if _QMult is not None:
            if isinstance(_QMult, ARRAY_LIKE):
                if _length_NPts is None:
                    _length_NPts = len(_QMult)
                    output.write(f' NPts={_length_NPts}')
                elif len(_QMult) != _length_NPts:
                    raise ValueError(f'Array length ({len(_QMult)}) for "QMult" does not match expected length ({_length_NPts})')

                _QMult = _as_list(_QMult)
            else:
                _length_QMult, _QMult = _filepath_array(_QMult)
                if _length_NPts is None:
                    _length_NPts = _length_QMult
                    output.write(f' NPts={_length_NPts}')
                elif _length_NPts != _length_QMult:
                    raise ValueError(f'Array length ({_length_QMult}) for "QMult" (from file) does not match expected length ({_length_NPts})')

            output.write(f' QMult={_QMult}')

        _UseActual = fields.get('UseActual')
        if _UseActual is not None:
            output.write(f' UseActual={_UseActual}')

        _PMax = fields.get('PMax')
        if _PMax is not None:
            output.write(f' PMax={_PMax}')

        _QMax = fields.get('QMax')
        if _QMax is not None:
            output.write(f' QMax={_QMax}')

        _PBase = fields.get('PBase')
        if _PBase is not None:
            output.write(f' PBase={_PBase}')

        _QBase = fields.get('QBase')
        if _QBase is not None:
            output.write(f' QBase={_QBase}')

        _PMult = fields.get('PMult')
        if _PMult is not None:
            if isinstance(_PMult, ARRAY_LIKE):
                if _length_NPts is None:
                    _length_NPts = len(_PMult)
                    output.write(f' NPts={_length_NPts}')
                elif len(_PMult) != _length_NPts:
                    raise ValueError(f'Array length ({len(_PMult)}) for "PMult" does not match expected length ({_length_NPts})')

                _PMult = _as_list(_PMult)
            else:
                _length_PMult, _PMult = _filepath_array(_PMult)
                if _length_NPts is None:
                    _length_NPts = _length_PMult
                    output.write(f' NPts={_length_NPts}')
                elif _length_NPts != _length_PMult:
                    raise ValueError(f'Array length ({_length_PMult}) for "PMult" (from file) does not match expected length ({_length_NPts})')

            output.write(f' PMult={_PMult}')

        _PQCSVFile = fields.get('PQCSVFile')
        if _PQCSVFile is not None:
            _length_NPts = _pqcsvfile_array_length(_PQCSVFile, cols=2 if (fields.get('Interval') or 0.0) == 0.0 else 1)
            output.write(f' NPts={_length_NPts}')
            output.write(f' PQCSVFile={_quoted(_PQCSVFile)}')

        _Interpolation = fields.get('Interpolation')
        if _Interpolation is not None:
            output.write(f' Interpolation={_quoted(_Interpolation)}')

        _Action = fields.get('Action')
        if _Action is not None:
            output.write(f' Action={_quoted(_Action)}')

        output.write('\n')

    @model_validator(mode="before")
    def _val_dss_model(self) -> "LoadShape":
        _fields_set = set(self.keys())
        # Validate oneOf (spec. sets) based on the `required` lists
        _required_PMultQMultHour = _fields_set.issuperset({'Hour', 'PMult'})
        _required_PMultQMultInterval = _fields_set.issuperset({'Interval', 'PMult'})
        _required_PQCSVFile = _fields_set.issuperset({'PQCSVFile'})
        _required_CSVFile = _fields_set.issuperset({'CSVFile'})
        _required_SngFile = _fields_set.issuperset({'SngFile'})
        _required_DblFile = _fields_set.issuperset({'DblFile'})
        num_specs = _required_PMultQMultHour + _required_PMultQMultInterval + _required_PQCSVFile + _required_CSVFile + _required_SngFile + _required_DblFile
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



LoadShape_ = LoadShape


class LoadShapeList(RootModel[List[LoadShape]]):
    root: List[LoadShape]





class LoadShapeContainer(RootModel[Union[LoadShapeList, JSONFilePath, JSONLinesFilePath]]):
    root: Union[LoadShapeList, JSONFilePath, JSONLinesFilePath] = Field(..., title="LoadShapeContainer")


    @model_validator(mode="before")
    def _val_dss_model(self) -> "LoadShapeContainer":
        try:
            _fields_set = set(self.keys())
        except:
            if isinstance(self, (list, tuple)):
                return self

            raise

        # Validate oneOf (spec. sets) based on the `required` lists
        _required_LoadShapeList = "root" in _fields_set and isinstance(self.root, LoadShapeList)
        _required_JSONFilePath = _fields_set.issuperset({'JSONFile'})
        _required_JSONLinesFilePath = _fields_set.issuperset({'JSONLinesFile'})
        num_specs = _required_LoadShapeList + _required_JSONFilePath + _required_JSONLinesFilePath
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



class TShape_Common(BaseModel):
    Name: Annotated[str, Field(max_length=255, min_length=1)] = Field(..., title="Name")
    Like: Optional[str] = Field(None, title="Like")
    Mean: Optional[float] = Field(None, title="Mean")
    StdDev: Optional[float] = Field(None, title="StdDev")
    Action: Optional[TShapeAction] = Field(None, title="Action")

class TShape_TempHour(TShape_Common):
    Temp: ArrayOrFilePath = Field(..., title="Temp")
    Hour: ArrayOrFilePath = Field(..., title="Hour")

class TShape_TempInterval(TShape_Common):
    Interval: Annotated[float, Field(ge=0)] = Field(..., title="Interval")
    Temp: ArrayOrFilePath = Field(..., title="Temp")

class TShape_CSVFile(TShape_Common):
    CSVFile: FilePath = Field(..., title="CSVFile")

class TShape_SngFile(TShape_Common):
    SngFile: FilePath = Field(..., title="SngFile")

class TShape_DblFile(TShape_Common):
    DblFile: FilePath = Field(..., title="DblFile")


class TShape(RootModel[Union[TShape_TempHour, TShape_TempInterval, TShape_CSVFile, TShape_SngFile, TShape_DblFile]]):
    root: Union[TShape_TempHour, TShape_TempInterval, TShape_CSVFile, TShape_SngFile, TShape_DblFile] = Field(..., title="TShape")

    def dumps_dss(self, edit=False) -> str:
        """
        Dump the DSS object dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            self.dump_dss(output, edit)
            return output.getvalue()

    def dump_dss(self, output: TextIO, edit=False):
        """
        Dump the DSS object to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        TShape.dict_dump_dss(self.model_dump(exclude_unset=True), output, edit=edit)

    @staticmethod
    def dict_dumps_dss(fields: Dict, edit=False) -> str:
        """
        Dump the DSS object from a dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            TShape.dict_dump_dss(fields, output, edit)
            return output.getvalue()

    @staticmethod
    def dict_dump_dss(fields: Dict, output: TextIO, edit=False):
        """
        Dump the DSS object from a dict to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        _length_NPts = None

        if edit:
            output.write(f'''edit TShape.{fields['Name']}''')
        else:
            output.write(f'''new TShape.{fields['Name']}''')

        _Like = fields.get('Like')
        if _Like is not None:
            output.write(f' Like={_quoted(_Like)}')

        _Interval = fields.get('Interval')
        if _Interval is not None:
            output.write(f' Interval={_Interval}')

        _Temp = fields.get('Temp')
        if _Temp is not None:
            if isinstance(_Temp, ARRAY_LIKE):
                _length_NPts = len(_Temp)
                output.write(f' NPts={_length_NPts}')
                _Temp = _as_list(_Temp)
            else:
                _length_Temp, _Temp = _filepath_array(_Temp)
                if _length_NPts is None:
                    _length_NPts = _length_Temp
                    output.write(f' NPts={_length_NPts}')
                elif _length_NPts != _length_Temp:
                    raise ValueError(f'Array length ({_length_Temp}) for "Temp" (from file) does not match expected length ({_length_NPts})')

            output.write(f' Temp={_Temp}')

        _Hour = fields.get('Hour')
        if _Hour is not None:
            if isinstance(_Hour, ARRAY_LIKE):
                if _length_NPts is None:
                    _length_NPts = len(_Hour)
                    output.write(f' NPts={_length_NPts}')
                elif len(_Hour) != _length_NPts:
                    raise ValueError(f'Array length ({len(_Hour)}) for "Hour" does not match expected length ({_length_NPts})')

                _Hour = _as_list(_Hour)
            else:
                _length_Hour, _Hour = _filepath_array(_Hour)
                if _length_NPts is None:
                    _length_NPts = _length_Hour
                    output.write(f' NPts={_length_NPts}')
                elif _length_NPts != _length_Hour:
                    raise ValueError(f'Array length ({_length_Hour}) for "Hour" (from file) does not match expected length ({_length_NPts})')

            output.write(f' Hour={_Hour}')

        _Mean = fields.get('Mean')
        if _Mean is not None:
            output.write(f' Mean={_Mean}')

        _StdDev = fields.get('StdDev')
        if _StdDev is not None:
            output.write(f' StdDev={_StdDev}')

        _CSVFile = fields.get('CSVFile')
        if _CSVFile is not None:
            _length_NPts = _csvfile_array_length(_CSVFile)
            output.write(f' NPts={_length_NPts}')
            output.write(f' CSVFile={_quoted(_CSVFile)}')

        _SngFile = fields.get('SngFile')
        if _SngFile is not None:
            _length_NPts = _sngfile_array_length(_SngFile, cols=2 if (fields.get('Interval') or 0.0) == 0.0 else 1)
            output.write(f' NPts={_length_NPts}')
            output.write(f' SngFile={_quoted(_SngFile)}')

        _DblFile = fields.get('DblFile')
        if _DblFile is not None:
            _length_NPts = _dblfile_array_length(_DblFile, cols=2 if (fields.get('Interval') or 0.0) == 0.0 else 1)
            output.write(f' NPts={_length_NPts}')
            output.write(f' DblFile={_quoted(_DblFile)}')

        _Action = fields.get('Action')
        if _Action is not None:
            output.write(f' Action={_quoted(_Action)}')

        output.write('\n')

    @model_validator(mode="before")
    def _val_dss_model(self) -> "TShape":
        _fields_set = set(self.keys())
        # Validate oneOf (spec. sets) based on the `required` lists
        _required_TempHour = _fields_set.issuperset({'Hour', 'Temp'})
        _required_TempInterval = _fields_set.issuperset({'Interval', 'Temp'})
        _required_CSVFile = _fields_set.issuperset({'CSVFile'})
        _required_SngFile = _fields_set.issuperset({'SngFile'})
        _required_DblFile = _fields_set.issuperset({'DblFile'})
        num_specs = _required_TempHour + _required_TempInterval + _required_CSVFile + _required_SngFile + _required_DblFile
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



TShape_ = TShape


class TShapeList(RootModel[List[TShape]]):
    root: List[TShape]





class TShapeContainer(RootModel[Union[TShapeList, JSONFilePath, JSONLinesFilePath]]):
    root: Union[TShapeList, JSONFilePath, JSONLinesFilePath] = Field(..., title="TShapeContainer")


    @model_validator(mode="before")
    def _val_dss_model(self) -> "TShapeContainer":
        try:
            _fields_set = set(self.keys())
        except:
            if isinstance(self, (list, tuple)):
                return self

            raise

        # Validate oneOf (spec. sets) based on the `required` lists
        _required_TShapeList = "root" in _fields_set and isinstance(self.root, TShapeList)
        _required_JSONFilePath = _fields_set.issuperset({'JSONFile'})
        _required_JSONLinesFilePath = _fields_set.issuperset({'JSONLinesFile'})
        num_specs = _required_TShapeList + _required_JSONFilePath + _required_JSONLinesFilePath
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



class PriceShape_Common(BaseModel):
    Name: Annotated[str, Field(max_length=255, min_length=1)] = Field(..., title="Name")
    Like: Optional[str] = Field(None, title="Like")
    Mean: Optional[float] = Field(None, title="Mean")
    StdDev: Optional[float] = Field(None, title="StdDev")
    Action: Optional[PriceShapeAction] = Field(None, title="Action")

class PriceShape_PriceHour(PriceShape_Common):
    Price: ArrayOrFilePath = Field(..., title="Price")
    Hour: ArrayOrFilePath = Field(..., title="Hour")

class PriceShape_PriceInterval(PriceShape_Common):
    Interval: float = Field(..., title="Interval")
    Price: ArrayOrFilePath = Field(..., title="Price")

class PriceShape_CSVFile(PriceShape_Common):
    CSVFile: FilePath = Field(..., title="CSVFile")

class PriceShape_SngFile(PriceShape_Common):
    SngFile: FilePath = Field(..., title="SngFile")

class PriceShape_DblFile(PriceShape_Common):
    DblFile: FilePath = Field(..., title="DblFile")


class PriceShape(RootModel[Union[PriceShape_PriceHour, PriceShape_PriceInterval, PriceShape_CSVFile, PriceShape_SngFile, PriceShape_DblFile]]):
    root: Union[PriceShape_PriceHour, PriceShape_PriceInterval, PriceShape_CSVFile, PriceShape_SngFile, PriceShape_DblFile] = Field(..., title="PriceShape")

    def dumps_dss(self, edit=False) -> str:
        """
        Dump the DSS object dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            self.dump_dss(output, edit)
            return output.getvalue()

    def dump_dss(self, output: TextIO, edit=False):
        """
        Dump the DSS object to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        PriceShape.dict_dump_dss(self.model_dump(exclude_unset=True), output, edit=edit)

    @staticmethod
    def dict_dumps_dss(fields: Dict, edit=False) -> str:
        """
        Dump the DSS object from a dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            PriceShape.dict_dump_dss(fields, output, edit)
            return output.getvalue()

    @staticmethod
    def dict_dump_dss(fields: Dict, output: TextIO, edit=False):
        """
        Dump the DSS object from a dict to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        _length_NPts = None

        if edit:
            output.write(f'''edit PriceShape.{fields['Name']}''')
        else:
            output.write(f'''new PriceShape.{fields['Name']}''')

        _Like = fields.get('Like')
        if _Like is not None:
            output.write(f' Like={_quoted(_Like)}')

        _Interval = fields.get('Interval')
        if _Interval is not None:
            output.write(f' Interval={_Interval}')

        _Price = fields.get('Price')
        if _Price is not None:
            if isinstance(_Price, ARRAY_LIKE):
                _length_NPts = len(_Price)
                output.write(f' NPts={_length_NPts}')
                _Price = _as_list(_Price)
            else:
                _length_Price, _Price = _filepath_array(_Price)
                if _length_NPts is None:
                    _length_NPts = _length_Price
                    output.write(f' NPts={_length_NPts}')
                elif _length_NPts != _length_Price:
                    raise ValueError(f'Array length ({_length_Price}) for "Price" (from file) does not match expected length ({_length_NPts})')

            output.write(f' Price={_Price}')

        _Hour = fields.get('Hour')
        if _Hour is not None:
            if isinstance(_Hour, ARRAY_LIKE):
                if _length_NPts is None:
                    _length_NPts = len(_Hour)
                    output.write(f' NPts={_length_NPts}')
                elif len(_Hour) != _length_NPts:
                    raise ValueError(f'Array length ({len(_Hour)}) for "Hour" does not match expected length ({_length_NPts})')

                _Hour = _as_list(_Hour)
            else:
                _length_Hour, _Hour = _filepath_array(_Hour)
                if _length_NPts is None:
                    _length_NPts = _length_Hour
                    output.write(f' NPts={_length_NPts}')
                elif _length_NPts != _length_Hour:
                    raise ValueError(f'Array length ({_length_Hour}) for "Hour" (from file) does not match expected length ({_length_NPts})')

            output.write(f' Hour={_Hour}')

        _Mean = fields.get('Mean')
        if _Mean is not None:
            output.write(f' Mean={_Mean}')

        _StdDev = fields.get('StdDev')
        if _StdDev is not None:
            output.write(f' StdDev={_StdDev}')

        _CSVFile = fields.get('CSVFile')
        if _CSVFile is not None:
            _length_NPts = _csvfile_array_length(_CSVFile)
            output.write(f' NPts={_length_NPts}')
            output.write(f' CSVFile={_quoted(_CSVFile)}')

        _SngFile = fields.get('SngFile')
        if _SngFile is not None:
            _length_NPts = _sngfile_array_length(_SngFile, cols=2 if (fields.get('Interval') or 0.0) == 0.0 else 1)
            output.write(f' NPts={_length_NPts}')
            output.write(f' SngFile={_quoted(_SngFile)}')

        _DblFile = fields.get('DblFile')
        if _DblFile is not None:
            _length_NPts = _dblfile_array_length(_DblFile, cols=2 if (fields.get('Interval') or 0.0) == 0.0 else 1)
            output.write(f' NPts={_length_NPts}')
            output.write(f' DblFile={_quoted(_DblFile)}')

        _Action = fields.get('Action')
        if _Action is not None:
            output.write(f' Action={_quoted(_Action)}')

        output.write('\n')

    @model_validator(mode="before")
    def _val_dss_model(self) -> "PriceShape":
        _fields_set = set(self.keys())
        # Validate oneOf (spec. sets) based on the `required` lists
        _required_PriceHour = _fields_set.issuperset({'Hour', 'Price'})
        _required_PriceInterval = _fields_set.issuperset({'Interval', 'Price'})
        _required_CSVFile = _fields_set.issuperset({'CSVFile'})
        _required_SngFile = _fields_set.issuperset({'SngFile'})
        _required_DblFile = _fields_set.issuperset({'DblFile'})
        num_specs = _required_PriceHour + _required_PriceInterval + _required_CSVFile + _required_SngFile + _required_DblFile
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



PriceShape_ = PriceShape


class PriceShapeList(RootModel[List[PriceShape]]):
    root: List[PriceShape]





class PriceShapeContainer(RootModel[Union[PriceShapeList, JSONFilePath, JSONLinesFilePath]]):
    root: Union[PriceShapeList, JSONFilePath, JSONLinesFilePath] = Field(..., title="PriceShapeContainer")


    @model_validator(mode="before")
    def _val_dss_model(self) -> "PriceShapeContainer":
        try:
            _fields_set = set(self.keys())
        except:
            if isinstance(self, (list, tuple)):
                return self

            raise

        # Validate oneOf (spec. sets) based on the `required` lists
        _required_PriceShapeList = "root" in _fields_set and isinstance(self.root, PriceShapeList)
        _required_JSONFilePath = _fields_set.issuperset({'JSONFile'})
        _required_JSONLinesFilePath = _fields_set.issuperset({'JSONLinesFile'})
        num_specs = _required_PriceShapeList + _required_JSONFilePath + _required_JSONLinesFilePath
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



class XYcurve_Common(BaseModel):
    Name: Annotated[str, Field(max_length=255, min_length=1)] = Field(..., title="Name")
    Like: Optional[str] = Field(None, title="Like")
    XShift: Optional[float] = Field(None, title="XShift")
    YShift: Optional[float] = Field(None, title="YShift")
    XScale: Optional[float] = Field(None, title="XScale")
    YScale: Optional[float] = Field(None, title="YScale")

class XYcurve_XarrayYarray(XYcurve_Common):
    YArray: ArrayOrFilePath = Field(..., title="YArray")
    XArray: ArrayOrFilePath = Field(..., title="XArray")

class XYcurve_CSVFile(XYcurve_Common):
    CSVFile: FilePath = Field(..., title="CSVFile")

class XYcurve_SngFile(XYcurve_Common):
    SngFile: FilePath = Field(..., title="SngFile")

class XYcurve_DblFile(XYcurve_Common):
    DblFile: FilePath = Field(..., title="DblFile")


class XYcurve(RootModel[Union[XYcurve_XarrayYarray, XYcurve_CSVFile, XYcurve_SngFile, XYcurve_DblFile]]):
    root: Union[XYcurve_XarrayYarray, XYcurve_CSVFile, XYcurve_SngFile, XYcurve_DblFile] = Field(..., title="XYcurve")

    def dumps_dss(self, edit=False) -> str:
        """
        Dump the DSS object dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            self.dump_dss(output, edit)
            return output.getvalue()

    def dump_dss(self, output: TextIO, edit=False):
        """
        Dump the DSS object to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        XYcurve.dict_dump_dss(self.model_dump(exclude_unset=True), output, edit=edit)

    @staticmethod
    def dict_dumps_dss(fields: Dict, edit=False) -> str:
        """
        Dump the DSS object from a dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            XYcurve.dict_dump_dss(fields, output, edit)
            return output.getvalue()

    @staticmethod
    def dict_dump_dss(fields: Dict, output: TextIO, edit=False):
        """
        Dump the DSS object from a dict to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        _length_NPts = None

        if edit:
            output.write(f'''edit XYcurve.{fields['Name']}''')
        else:
            output.write(f'''new XYcurve.{fields['Name']}''')

        _Like = fields.get('Like')
        if _Like is not None:
            output.write(f' Like={_quoted(_Like)}')

        _YArray = fields.get('YArray')
        if _YArray is not None:
            if isinstance(_YArray, ARRAY_LIKE):
                _length_NPts = len(_YArray)
                output.write(f' NPts={_length_NPts}')
                _YArray = _as_list(_YArray)
            else:
                _length_YArray, _YArray = _filepath_array(_YArray)
                if _length_NPts is None:
                    _length_NPts = _length_YArray
                    output.write(f' NPts={_length_NPts}')
                elif _length_NPts != _length_YArray:
                    raise ValueError(f'Array length ({_length_YArray}) for "YArray" (from file) does not match expected length ({_length_NPts})')

            output.write(f' YArray={_YArray}')

        _XArray = fields.get('XArray')
        if _XArray is not None:
            if isinstance(_XArray, ARRAY_LIKE):
                if _length_NPts is None:
                    _length_NPts = len(_XArray)
                    output.write(f' NPts={_length_NPts}')
                elif len(_XArray) != _length_NPts:
                    raise ValueError(f'Array length ({len(_XArray)}) for "XArray" does not match expected length ({_length_NPts})')

                _XArray = _as_list(_XArray)
            else:
                _length_XArray, _XArray = _filepath_array(_XArray)
                if _length_NPts is None:
                    _length_NPts = _length_XArray
                    output.write(f' NPts={_length_NPts}')
                elif _length_NPts != _length_XArray:
                    raise ValueError(f'Array length ({_length_XArray}) for "XArray" (from file) does not match expected length ({_length_NPts})')

            output.write(f' XArray={_XArray}')

        _CSVFile = fields.get('CSVFile')
        if _CSVFile is not None:
            _length_NPts = _csvfile_array_length(_CSVFile)
            output.write(f' NPts={_length_NPts}')
            output.write(f' CSVFile={_quoted(_CSVFile)}')

        _SngFile = fields.get('SngFile')
        if _SngFile is not None:
            _length_NPts = _sngfile_array_length(_SngFile)
            output.write(f' NPts={_length_NPts}')
            output.write(f' SngFile={_quoted(_SngFile)}')

        _DblFile = fields.get('DblFile')
        if _DblFile is not None:
            _length_NPts = _dblfile_array_length(_DblFile)
            output.write(f' NPts={_length_NPts}')
            output.write(f' DblFile={_quoted(_DblFile)}')

        _XShift = fields.get('XShift')
        if _XShift is not None:
            output.write(f' XShift={_XShift}')

        _YShift = fields.get('YShift')
        if _YShift is not None:
            output.write(f' YShift={_YShift}')

        _XScale = fields.get('XScale')
        if _XScale is not None:
            output.write(f' XScale={_XScale}')

        _YScale = fields.get('YScale')
        if _YScale is not None:
            output.write(f' YScale={_YScale}')

        output.write('\n')

    @model_validator(mode="before")
    def _val_dss_model(self) -> "XYcurve":
        _fields_set = set(self.keys())
        # Validate oneOf (spec. sets) based on the `required` lists
        _required_XarrayYarray = _fields_set.issuperset({'XArray', 'YArray'})
        _required_CSVFile = _fields_set.issuperset({'CSVFile'})
        _required_SngFile = _fields_set.issuperset({'SngFile'})
        _required_DblFile = _fields_set.issuperset({'DblFile'})
        num_specs = _required_XarrayYarray + _required_CSVFile + _required_SngFile + _required_DblFile
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



XYcurve_ = XYcurve


class XYcurveList(RootModel[List[XYcurve]]):
    root: List[XYcurve]





class XYcurveContainer(RootModel[Union[XYcurveList, JSONFilePath, JSONLinesFilePath]]):
    root: Union[XYcurveList, JSONFilePath, JSONLinesFilePath] = Field(..., title="XYcurveContainer")


    @model_validator(mode="before")
    def _val_dss_model(self) -> "XYcurveContainer":
        try:
            _fields_set = set(self.keys())
        except:
            if isinstance(self, (list, tuple)):
                return self

            raise

        # Validate oneOf (spec. sets) based on the `required` lists
        _required_XYcurveList = "root" in _fields_set and isinstance(self.root, XYcurveList)
        _required_JSONFilePath = _fields_set.issuperset({'JSONFile'})
        _required_JSONLinesFilePath = _fields_set.issuperset({'JSONLinesFile'})
        num_specs = _required_XYcurveList + _required_JSONFilePath + _required_JSONLinesFilePath
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



class GrowthShape_Common(BaseModel):
    Name: Annotated[str, Field(max_length=255, min_length=1)] = Field(..., title="Name")
    Like: Optional[str] = Field(None, title="Like")

class GrowthShape_YearMult(GrowthShape_Common):
    Year: ArrayOrFilePath = Field(..., title="Year")
    Mult: ArrayOrFilePath = Field(..., title="Mult")

class GrowthShape_CSVFile(GrowthShape_Common):
    CSVFile: FilePath = Field(..., title="CSVFile")

class GrowthShape_SngFile(GrowthShape_Common):
    SngFile: FilePath = Field(..., title="SngFile")

class GrowthShape_DblFile(GrowthShape_Common):
    DblFile: FilePath = Field(..., title="DblFile")


class GrowthShape(RootModel[Union[GrowthShape_YearMult, GrowthShape_CSVFile, GrowthShape_SngFile, GrowthShape_DblFile]]):
    root: Union[GrowthShape_YearMult, GrowthShape_CSVFile, GrowthShape_SngFile, GrowthShape_DblFile] = Field(..., title="GrowthShape")

    def dumps_dss(self, edit=False) -> str:
        """
        Dump the DSS object dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            self.dump_dss(output, edit)
            return output.getvalue()

    def dump_dss(self, output: TextIO, edit=False):
        """
        Dump the DSS object to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        GrowthShape.dict_dump_dss(self.model_dump(exclude_unset=True), output, edit=edit)

    @staticmethod
    def dict_dumps_dss(fields: Dict, edit=False) -> str:
        """
        Dump the DSS object from a dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            GrowthShape.dict_dump_dss(fields, output, edit)
            return output.getvalue()

    @staticmethod
    def dict_dump_dss(fields: Dict, output: TextIO, edit=False):
        """
        Dump the DSS object from a dict to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        _length_NPts = None

        if edit:
            output.write(f'''edit GrowthShape.{fields['Name']}''')
        else:
            output.write(f'''new GrowthShape.{fields['Name']}''')

        _Like = fields.get('Like')
        if _Like is not None:
            output.write(f' Like={_quoted(_Like)}')

        _Year = fields.get('Year')
        if _Year is not None:
            if isinstance(_Year, ARRAY_LIKE):
                _length_NPts = len(_Year)
                output.write(f' NPts={_length_NPts}')
                _Year = _as_list(_Year)
            else:
                _length_Year, _Year = _filepath_array(_Year)
                if _length_NPts is None:
                    _length_NPts = _length_Year
                    output.write(f' NPts={_length_NPts}')
                elif _length_NPts != _length_Year:
                    raise ValueError(f'Array length ({_length_Year}) for "Year" (from file) does not match expected length ({_length_NPts})')

            output.write(f' Year={_Year}')

        _Mult = fields.get('Mult')
        if _Mult is not None:
            if isinstance(_Mult, ARRAY_LIKE):
                if _length_NPts is None:
                    _length_NPts = len(_Mult)
                    output.write(f' NPts={_length_NPts}')
                elif len(_Mult) != _length_NPts:
                    raise ValueError(f'Array length ({len(_Mult)}) for "Mult" does not match expected length ({_length_NPts})')

                _Mult = _as_list(_Mult)
            else:
                _length_Mult, _Mult = _filepath_array(_Mult)
                if _length_NPts is None:
                    _length_NPts = _length_Mult
                    output.write(f' NPts={_length_NPts}')
                elif _length_NPts != _length_Mult:
                    raise ValueError(f'Array length ({_length_Mult}) for "Mult" (from file) does not match expected length ({_length_NPts})')

            output.write(f' Mult={_Mult}')

        _CSVFile = fields.get('CSVFile')
        if _CSVFile is not None:
            _length_NPts = _csvfile_array_length(_CSVFile)
            output.write(f' NPts={_length_NPts}')
            output.write(f' CSVFile={_quoted(_CSVFile)}')

        _SngFile = fields.get('SngFile')
        if _SngFile is not None:
            _length_NPts = _sngfile_array_length(_SngFile)
            output.write(f' NPts={_length_NPts}')
            output.write(f' SngFile={_quoted(_SngFile)}')

        _DblFile = fields.get('DblFile')
        if _DblFile is not None:
            _length_NPts = _dblfile_array_length(_DblFile)
            output.write(f' NPts={_length_NPts}')
            output.write(f' DblFile={_quoted(_DblFile)}')

        output.write('\n')

    @model_validator(mode="before")
    def _val_dss_model(self) -> "GrowthShape":
        _fields_set = set(self.keys())
        # Validate oneOf (spec. sets) based on the `required` lists
        _required_YearMult = _fields_set.issuperset({'Mult', 'Year'})
        _required_CSVFile = _fields_set.issuperset({'CSVFile'})
        _required_SngFile = _fields_set.issuperset({'SngFile'})
        _required_DblFile = _fields_set.issuperset({'DblFile'})
        num_specs = _required_YearMult + _required_CSVFile + _required_SngFile + _required_DblFile
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



GrowthShape_ = GrowthShape


class GrowthShapeList(RootModel[List[GrowthShape]]):
    root: List[GrowthShape]





class GrowthShapeContainer(RootModel[Union[GrowthShapeList, JSONFilePath, JSONLinesFilePath]]):
    root: Union[GrowthShapeList, JSONFilePath, JSONLinesFilePath] = Field(..., title="GrowthShapeContainer")


    @model_validator(mode="before")
    def _val_dss_model(self) -> "GrowthShapeContainer":
        try:
            _fields_set = set(self.keys())
        except:
            if isinstance(self, (list, tuple)):
                return self

            raise

        # Validate oneOf (spec. sets) based on the `required` lists
        _required_GrowthShapeList = "root" in _fields_set and isinstance(self.root, GrowthShapeList)
        _required_JSONFilePath = _fields_set.issuperset({'JSONFile'})
        _required_JSONLinesFilePath = _fields_set.issuperset({'JSONLinesFile'})
        num_specs = _required_GrowthShapeList + _required_JSONFilePath + _required_JSONLinesFilePath
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



class TCC_Curve(BaseModel):
    Name: Annotated[str, Field(max_length=255, min_length=1)] = Field(..., title="Name")
    Like: Optional[str] = Field(None, title="Like")
    C_Array: Optional[ArrayOrFilePath] = Field(None, title="C_Array")
    T_Array: Optional[ArrayOrFilePath] = Field(None, title="T_Array")

    def dumps_dss(self, edit=False) -> str:
        """
        Dump the DSS object dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            self.dump_dss(output, edit)
            return output.getvalue()

    def dump_dss(self, output: TextIO, edit=False):
        """
        Dump the DSS object to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        TCC_Curve.dict_dump_dss(self.model_dump(exclude_unset=True), output, edit=edit)

    @staticmethod
    def dict_dumps_dss(fields: Dict, edit=False) -> str:
        """
        Dump the DSS object from a dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            TCC_Curve.dict_dump_dss(fields, output, edit)
            return output.getvalue()

    @staticmethod
    def dict_dump_dss(fields: Dict, output: TextIO, edit=False):
        """
        Dump the DSS object from a dict to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        _length_NPts = None

        if edit:
            output.write(f'''edit TCC_Curve.{fields['Name']}''')
        else:
            output.write(f'''new TCC_Curve.{fields['Name']}''')

        _Like = fields.get('Like')
        if _Like is not None:
            output.write(f' Like={_quoted(_Like)}')

        _C_Array = fields.get('C_Array')
        if _C_Array is not None:
            if isinstance(_C_Array, ARRAY_LIKE):
                _length_NPts = len(_C_Array)
                output.write(f' NPts={_length_NPts}')
                _C_Array = _as_list(_C_Array)
            else:
                _length_C_Array, _C_Array = _filepath_array(_C_Array)
                if _length_NPts is None:
                    _length_NPts = _length_C_Array
                    output.write(f' NPts={_length_NPts}')
                elif _length_NPts != _length_C_Array:
                    raise ValueError(f'Array length ({_length_C_Array}) for "C_Array" (from file) does not match expected length ({_length_NPts})')

            output.write(f' C_Array={_C_Array}')

        _T_Array = fields.get('T_Array')
        if _T_Array is not None:
            if isinstance(_T_Array, ARRAY_LIKE):
                if _length_NPts is None:
                    _length_NPts = len(_T_Array)
                    output.write(f' NPts={_length_NPts}')
                elif len(_T_Array) != _length_NPts:
                    raise ValueError(f'Array length ({len(_T_Array)}) for "T_Array" does not match expected length ({_length_NPts})')

                _T_Array = _as_list(_T_Array)
            else:
                _length_T_Array, _T_Array = _filepath_array(_T_Array)
                if _length_NPts is None:
                    _length_NPts = _length_T_Array
                    output.write(f' NPts={_length_NPts}')
                elif _length_NPts != _length_T_Array:
                    raise ValueError(f'Array length ({_length_T_Array}) for "T_Array" (from file) does not match expected length ({_length_NPts})')

            output.write(f' T_Array={_T_Array}')

        output.write('\n')


TCC_Curve_ = TCC_Curve


class TCC_CurveList(RootModel[List[TCC_Curve]]):
    root: List[TCC_Curve]





class TCC_CurveContainer(RootModel[Union[TCC_CurveList, JSONFilePath, JSONLinesFilePath]]):
    root: Union[TCC_CurveList, JSONFilePath, JSONLinesFilePath] = Field(..., title="TCC_CurveContainer")


    @model_validator(mode="before")
    def _val_dss_model(self) -> "TCC_CurveContainer":
        try:
            _fields_set = set(self.keys())
        except:
            if isinstance(self, (list, tuple)):
                return self

            raise

        # Validate oneOf (spec. sets) based on the `required` lists
        _required_TCC_CurveList = "root" in _fields_set and isinstance(self.root, TCC_CurveList)
        _required_JSONFilePath = _fields_set.issuperset({'JSONFile'})
        _required_JSONLinesFilePath = _fields_set.issuperset({'JSONLinesFile'})
        num_specs = _required_TCC_CurveList + _required_JSONFilePath + _required_JSONLinesFilePath
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



class Spectrum_Common(BaseModel):
    Name: Annotated[str, Field(max_length=255, min_length=1)] = Field(..., title="Name")
    Like: Optional[str] = Field(None, title="Like")

class Spectrum_HarmonicAnglepctMag(Spectrum_Common):
    Harmonic: ArrayOrFilePath = Field(..., title="Harmonic")
    pctMag: ArrayOrFilePath = Field(..., title="%Mag", validation_alias=AliasChoices("pctMag", "%Mag"))
    Angle: ArrayOrFilePath = Field(..., title="Angle")

class Spectrum_CSVFile(Spectrum_Common):
    CSVFile: FilePath = Field(..., title="CSVFile")


class Spectrum(RootModel[Union[Spectrum_HarmonicAnglepctMag, Spectrum_CSVFile]]):
    root: Union[Spectrum_HarmonicAnglepctMag, Spectrum_CSVFile] = Field(..., title="Spectrum")

    def dumps_dss(self, edit=False) -> str:
        """
        Dump the DSS object dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            self.dump_dss(output, edit)
            return output.getvalue()

    def dump_dss(self, output: TextIO, edit=False):
        """
        Dump the DSS object to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        Spectrum.dict_dump_dss(self.model_dump(exclude_unset=True), output, edit=edit)

    @staticmethod
    def dict_dumps_dss(fields: Dict, edit=False) -> str:
        """
        Dump the DSS object from a dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            Spectrum.dict_dump_dss(fields, output, edit)
            return output.getvalue()

    @staticmethod
    def dict_dump_dss(fields: Dict, output: TextIO, edit=False):
        """
        Dump the DSS object from a dict to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        _length_NumHarm = None

        if edit:
            output.write(f'''edit Spectrum.{fields['Name']}''')
        else:
            output.write(f'''new Spectrum.{fields['Name']}''')

        _Like = fields.get('Like')
        if _Like is not None:
            output.write(f' Like={_quoted(_Like)}')

        _Harmonic = fields.get('Harmonic')
        if _Harmonic is not None:
            if isinstance(_Harmonic, ARRAY_LIKE):
                _length_NumHarm = len(_Harmonic)
                output.write(f' NumHarm={_length_NumHarm}')
                _Harmonic = _as_list(_Harmonic)
            else:
                _length_Harmonic, _Harmonic = _filepath_array(_Harmonic)
                if _length_NumHarm is None:
                    _length_NumHarm = _length_Harmonic
                    output.write(f' NumHarm={_length_NumHarm}')
                elif _length_NumHarm != _length_Harmonic:
                    raise ValueError(f'Array length ({_length_Harmonic}) for "Harmonic" (from file) does not match expected length ({_length_NumHarm})')

            output.write(f' Harmonic={_Harmonic}')

        _pctMag = fields.get('pctMag')
        if _pctMag is not None:
            if isinstance(_pctMag, ARRAY_LIKE):
                if _length_NumHarm is None:
                    _length_NumHarm = len(_pctMag)
                    output.write(f' NumHarm={_length_NumHarm}')
                elif len(_pctMag) != _length_NumHarm:
                    raise ValueError(f'Array length ({len(_pctMag)}) for "pctMag" does not match expected length ({_length_NumHarm})')

                _pctMag = _as_list(_pctMag)
            else:
                _length_pctMag, _pctMag = _filepath_array(_pctMag)
                if _length_NumHarm is None:
                    _length_NumHarm = _length_pctMag
                    output.write(f' NumHarm={_length_NumHarm}')
                elif _length_NumHarm != _length_pctMag:
                    raise ValueError(f'Array length ({_length_pctMag}) for "pctMag" (from file) does not match expected length ({_length_NumHarm})')

            output.write(f' %Mag={_pctMag}')

        _Angle = fields.get('Angle')
        if _Angle is not None:
            if isinstance(_Angle, ARRAY_LIKE):
                if _length_NumHarm is None:
                    _length_NumHarm = len(_Angle)
                    output.write(f' NumHarm={_length_NumHarm}')
                elif len(_Angle) != _length_NumHarm:
                    raise ValueError(f'Array length ({len(_Angle)}) for "Angle" does not match expected length ({_length_NumHarm})')

                _Angle = _as_list(_Angle)
            else:
                _length_Angle, _Angle = _filepath_array(_Angle)
                if _length_NumHarm is None:
                    _length_NumHarm = _length_Angle
                    output.write(f' NumHarm={_length_NumHarm}')
                elif _length_NumHarm != _length_Angle:
                    raise ValueError(f'Array length ({_length_Angle}) for "Angle" (from file) does not match expected length ({_length_NumHarm})')

            output.write(f' Angle={_Angle}')

        _CSVFile = fields.get('CSVFile')
        if _CSVFile is not None:
            _length_NumHarm = _csvfile_array_length(_CSVFile)
            output.write(f' NumHarm={_length_NumHarm}')
            output.write(f' CSVFile={_quoted(_CSVFile)}')

        output.write('\n')

    @model_validator(mode="before")
    def _val_dss_model(self) -> "Spectrum":
        _fields_set = set(self.keys())
        # Validate oneOf (spec. sets) based on the `required` lists
        _required_HarmonicAnglepctMag = _fields_set.issuperset({'Angle', 'Harmonic', 'pctMag'})
        _required_CSVFile = _fields_set.issuperset({'CSVFile'})
        num_specs = _required_HarmonicAnglepctMag + _required_CSVFile
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



Spectrum_ = Spectrum


class SpectrumList(RootModel[List[Spectrum]]):
    root: List[Spectrum]





class SpectrumContainer(RootModel[Union[SpectrumList, JSONFilePath, JSONLinesFilePath]]):
    root: Union[SpectrumList, JSONFilePath, JSONLinesFilePath] = Field(..., title="SpectrumContainer")


    @model_validator(mode="before")
    def _val_dss_model(self) -> "SpectrumContainer":
        try:
            _fields_set = set(self.keys())
        except:
            if isinstance(self, (list, tuple)):
                return self

            raise

        # Validate oneOf (spec. sets) based on the `required` lists
        _required_SpectrumList = "root" in _fields_set and isinstance(self.root, SpectrumList)
        _required_JSONFilePath = _fields_set.issuperset({'JSONFile'})
        _required_JSONLinesFilePath = _fields_set.issuperset({'JSONLinesFile'})
        num_specs = _required_SpectrumList + _required_JSONFilePath + _required_JSONLinesFilePath
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



class WireData(BaseModel):
    Name: Annotated[str, Field(max_length=255, min_length=1)] = Field(..., title="Name")
    Like: Optional[str] = Field(None, title="Like")
    RDC: Optional[float] = Field(None, title="RDC")
    RAC: Optional[float] = Field(None, title="RAC")
    RUnits: Optional[LengthUnit] = Field(None, title="RUnits")
    GMRAC: Optional[PositiveFloat] = Field(None, title="GMRAC")
    GMRUnits: Optional[LengthUnit] = Field(None, title="GMRUnits")
    Radius: Optional[PositiveFloat] = Field(None, title="Radius")
    RadUnits: Optional[LengthUnit] = Field(None, title="RadUnits")
    NormAmps: Optional[float] = Field(None, title="NormAmps")
    EmergAmps: Optional[float] = Field(None, title="EmergAmps")
    Ratings: Optional[ArrayOrFilePath] = Field(None, title="Ratings")
    CapRadius: Optional[PositiveFloat] = Field(None, title="CapRadius")

    def dumps_dss(self, edit=False) -> str:
        """
        Dump the DSS object dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            self.dump_dss(output, edit)
            return output.getvalue()

    def dump_dss(self, output: TextIO, edit=False):
        """
        Dump the DSS object to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        WireData.dict_dump_dss(self.model_dump(exclude_unset=True), output, edit=edit)

    @staticmethod
    def dict_dumps_dss(fields: Dict, edit=False) -> str:
        """
        Dump the DSS object from a dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            WireData.dict_dump_dss(fields, output, edit)
            return output.getvalue()

    @staticmethod
    def dict_dump_dss(fields: Dict, output: TextIO, edit=False):
        """
        Dump the DSS object from a dict to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        if edit:
            output.write(f'''edit WireData.{fields['Name']}''')
        else:
            output.write(f'''new WireData.{fields['Name']}''')

        _Like = fields.get('Like')
        if _Like is not None:
            output.write(f' Like={_quoted(_Like)}')

        _RDC = fields.get('RDC')
        if _RDC is not None:
            output.write(f' RDC={_RDC}')

        _RAC = fields.get('RAC')
        if _RAC is not None:
            output.write(f' RAC={_RAC}')

        _RUnits = fields.get('RUnits')
        if _RUnits is not None:
            output.write(f' RUnits={_quoted(_RUnits)}')

        _GMRAC = fields.get('GMRAC')
        if _GMRAC is not None:
            output.write(f' GMRAC={_GMRAC}')

        _GMRUnits = fields.get('GMRUnits')
        if _GMRUnits is not None:
            output.write(f' GMRUnits={_quoted(_GMRUnits)}')

        _Radius = fields.get('Radius')
        if _Radius is not None:
            output.write(f' Radius={_Radius}')

        _RadUnits = fields.get('RadUnits')
        if _RadUnits is not None:
            output.write(f' RadUnits={_quoted(_RadUnits)}')

        _NormAmps = fields.get('NormAmps')
        if _NormAmps is not None:
            output.write(f' NormAmps={_NormAmps}')

        _EmergAmps = fields.get('EmergAmps')
        if _EmergAmps is not None:
            output.write(f' EmergAmps={_EmergAmps}')

        _Ratings = fields.get('Ratings')
        if _Ratings is not None:
            if isinstance(_Ratings, ARRAY_LIKE):
                output.write(f' Seasons={len(_Ratings)}')
                _Ratings = _as_list(_Ratings)
            else:
                _length_Ratings, _Ratings = _filepath_array(_Ratings)
                output.write(f' Seasons={_length_Ratings}')
            output.write(f' Ratings={_Ratings}')

        _CapRadius = fields.get('CapRadius')
        if _CapRadius is not None:
            output.write(f' CapRadius={_CapRadius}')

        output.write('\n')


WireData_ = WireData


class WireDataList(RootModel[List[WireData]]):
    root: List[WireData]





class WireDataContainer(RootModel[Union[WireDataList, JSONFilePath, JSONLinesFilePath]]):
    root: Union[WireDataList, JSONFilePath, JSONLinesFilePath] = Field(..., title="WireDataContainer")


    @model_validator(mode="before")
    def _val_dss_model(self) -> "WireDataContainer":
        try:
            _fields_set = set(self.keys())
        except:
            if isinstance(self, (list, tuple)):
                return self

            raise

        # Validate oneOf (spec. sets) based on the `required` lists
        _required_WireDataList = "root" in _fields_set and isinstance(self.root, WireDataList)
        _required_JSONFilePath = _fields_set.issuperset({'JSONFile'})
        _required_JSONLinesFilePath = _fields_set.issuperset({'JSONLinesFile'})
        num_specs = _required_WireDataList + _required_JSONFilePath + _required_JSONLinesFilePath
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



class CNData(BaseModel):
    Name: Annotated[str, Field(max_length=255, min_length=1)] = Field(..., title="Name")
    Like: Optional[str] = Field(None, title="Like")
    k: Optional[int] = Field(None, title="k")
    DiaStrand: Optional[PositiveFloat] = Field(None, title="DiaStrand")
    GMRStrand: Optional[PositiveFloat] = Field(None, title="GMRStrand")
    RStrand: Optional[float] = Field(None, title="RStrand")
    EpsR: Optional[float] = Field(None, title="EpsR")
    InsLayer: Optional[PositiveFloat] = Field(None, title="InsLayer")
    DiaIns: Optional[PositiveFloat] = Field(None, title="DiaIns")
    DiaCable: Optional[PositiveFloat] = Field(None, title="DiaCable")
    RDC: Optional[float] = Field(None, title="RDC")
    RAC: Optional[float] = Field(None, title="RAC")
    RUnits: Optional[LengthUnit] = Field(None, title="RUnits")
    GMRAC: Optional[PositiveFloat] = Field(None, title="GMRAC")
    GMRUnits: Optional[LengthUnit] = Field(None, title="GMRUnits")
    Radius: Optional[PositiveFloat] = Field(None, title="Radius")
    RadUnits: Optional[LengthUnit] = Field(None, title="RadUnits")
    NormAmps: Optional[float] = Field(None, title="NormAmps")
    EmergAmps: Optional[float] = Field(None, title="EmergAmps")
    Ratings: Optional[ArrayOrFilePath] = Field(None, title="Ratings")
    CapRadius: Optional[PositiveFloat] = Field(None, title="CapRadius")

    def dumps_dss(self, edit=False) -> str:
        """
        Dump the DSS object dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            self.dump_dss(output, edit)
            return output.getvalue()

    def dump_dss(self, output: TextIO, edit=False):
        """
        Dump the DSS object to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        CNData.dict_dump_dss(self.model_dump(exclude_unset=True), output, edit=edit)

    @staticmethod
    def dict_dumps_dss(fields: Dict, edit=False) -> str:
        """
        Dump the DSS object from a dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            CNData.dict_dump_dss(fields, output, edit)
            return output.getvalue()

    @staticmethod
    def dict_dump_dss(fields: Dict, output: TextIO, edit=False):
        """
        Dump the DSS object from a dict to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        if edit:
            output.write(f'''edit CNData.{fields['Name']}''')
        else:
            output.write(f'''new CNData.{fields['Name']}''')

        _Like = fields.get('Like')
        if _Like is not None:
            output.write(f' Like={_quoted(_Like)}')

        _k = fields.get('k')
        if _k is not None:
            output.write(f' k={_k}')

        _DiaStrand = fields.get('DiaStrand')
        if _DiaStrand is not None:
            output.write(f' DiaStrand={_DiaStrand}')

        _GMRStrand = fields.get('GMRStrand')
        if _GMRStrand is not None:
            output.write(f' GMRStrand={_GMRStrand}')

        _RStrand = fields.get('RStrand')
        if _RStrand is not None:
            output.write(f' RStrand={_RStrand}')

        _EpsR = fields.get('EpsR')
        if _EpsR is not None:
            output.write(f' EpsR={_EpsR}')

        _InsLayer = fields.get('InsLayer')
        if _InsLayer is not None:
            output.write(f' InsLayer={_InsLayer}')

        _DiaIns = fields.get('DiaIns')
        if _DiaIns is not None:
            output.write(f' DiaIns={_DiaIns}')

        _DiaCable = fields.get('DiaCable')
        if _DiaCable is not None:
            output.write(f' DiaCable={_DiaCable}')

        _RDC = fields.get('RDC')
        if _RDC is not None:
            output.write(f' RDC={_RDC}')

        _RAC = fields.get('RAC')
        if _RAC is not None:
            output.write(f' RAC={_RAC}')

        _RUnits = fields.get('RUnits')
        if _RUnits is not None:
            output.write(f' RUnits={_quoted(_RUnits)}')

        _GMRAC = fields.get('GMRAC')
        if _GMRAC is not None:
            output.write(f' GMRAC={_GMRAC}')

        _GMRUnits = fields.get('GMRUnits')
        if _GMRUnits is not None:
            output.write(f' GMRUnits={_quoted(_GMRUnits)}')

        _Radius = fields.get('Radius')
        if _Radius is not None:
            output.write(f' Radius={_Radius}')

        _RadUnits = fields.get('RadUnits')
        if _RadUnits is not None:
            output.write(f' RadUnits={_quoted(_RadUnits)}')

        _NormAmps = fields.get('NormAmps')
        if _NormAmps is not None:
            output.write(f' NormAmps={_NormAmps}')

        _EmergAmps = fields.get('EmergAmps')
        if _EmergAmps is not None:
            output.write(f' EmergAmps={_EmergAmps}')

        _Ratings = fields.get('Ratings')
        if _Ratings is not None:
            if isinstance(_Ratings, ARRAY_LIKE):
                output.write(f' Seasons={len(_Ratings)}')
                _Ratings = _as_list(_Ratings)
            else:
                _length_Ratings, _Ratings = _filepath_array(_Ratings)
                output.write(f' Seasons={_length_Ratings}')
            output.write(f' Ratings={_Ratings}')

        _CapRadius = fields.get('CapRadius')
        if _CapRadius is not None:
            output.write(f' CapRadius={_CapRadius}')

        output.write('\n')


CNData_ = CNData


class CNDataList(RootModel[List[CNData]]):
    root: List[CNData]





class CNDataContainer(RootModel[Union[CNDataList, JSONFilePath, JSONLinesFilePath]]):
    root: Union[CNDataList, JSONFilePath, JSONLinesFilePath] = Field(..., title="CNDataContainer")


    @model_validator(mode="before")
    def _val_dss_model(self) -> "CNDataContainer":
        try:
            _fields_set = set(self.keys())
        except:
            if isinstance(self, (list, tuple)):
                return self

            raise

        # Validate oneOf (spec. sets) based on the `required` lists
        _required_CNDataList = "root" in _fields_set and isinstance(self.root, CNDataList)
        _required_JSONFilePath = _fields_set.issuperset({'JSONFile'})
        _required_JSONLinesFilePath = _fields_set.issuperset({'JSONLinesFile'})
        num_specs = _required_CNDataList + _required_JSONFilePath + _required_JSONLinesFilePath
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



class TSData(BaseModel):
    Name: Annotated[str, Field(max_length=255, min_length=1)] = Field(..., title="Name")
    Like: Optional[str] = Field(None, title="Like")
    DiaShield: Optional[PositiveFloat] = Field(None, title="DiaShield")
    TapeLayer: Optional[PositiveFloat] = Field(None, title="TapeLayer")
    TapeLap: Optional[Annotated[float, Field(ge=0)]] = Field(None, title="TapeLap")
    EpsR: Optional[float] = Field(None, title="EpsR")
    InsLayer: Optional[PositiveFloat] = Field(None, title="InsLayer")
    DiaIns: Optional[PositiveFloat] = Field(None, title="DiaIns")
    DiaCable: Optional[PositiveFloat] = Field(None, title="DiaCable")
    RDC: Optional[float] = Field(None, title="RDC")
    RAC: Optional[float] = Field(None, title="RAC")
    RUnits: Optional[LengthUnit] = Field(None, title="RUnits")
    GMRAC: Optional[PositiveFloat] = Field(None, title="GMRAC")
    GMRUnits: Optional[LengthUnit] = Field(None, title="GMRUnits")
    Radius: Optional[PositiveFloat] = Field(None, title="Radius")
    RadUnits: Optional[LengthUnit] = Field(None, title="RadUnits")
    NormAmps: Optional[float] = Field(None, title="NormAmps")
    EmergAmps: Optional[float] = Field(None, title="EmergAmps")
    Ratings: Optional[ArrayOrFilePath] = Field(None, title="Ratings")
    CapRadius: Optional[PositiveFloat] = Field(None, title="CapRadius")

    def dumps_dss(self, edit=False) -> str:
        """
        Dump the DSS object dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            self.dump_dss(output, edit)
            return output.getvalue()

    def dump_dss(self, output: TextIO, edit=False):
        """
        Dump the DSS object to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        TSData.dict_dump_dss(self.model_dump(exclude_unset=True), output, edit=edit)

    @staticmethod
    def dict_dumps_dss(fields: Dict, edit=False) -> str:
        """
        Dump the DSS object from a dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            TSData.dict_dump_dss(fields, output, edit)
            return output.getvalue()

    @staticmethod
    def dict_dump_dss(fields: Dict, output: TextIO, edit=False):
        """
        Dump the DSS object from a dict to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        if edit:
            output.write(f'''edit TSData.{fields['Name']}''')
        else:
            output.write(f'''new TSData.{fields['Name']}''')

        _Like = fields.get('Like')
        if _Like is not None:
            output.write(f' Like={_quoted(_Like)}')

        _DiaShield = fields.get('DiaShield')
        if _DiaShield is not None:
            output.write(f' DiaShield={_DiaShield}')

        _TapeLayer = fields.get('TapeLayer')
        if _TapeLayer is not None:
            output.write(f' TapeLayer={_TapeLayer}')

        _TapeLap = fields.get('TapeLap')
        if _TapeLap is not None:
            output.write(f' TapeLap={_TapeLap}')

        _EpsR = fields.get('EpsR')
        if _EpsR is not None:
            output.write(f' EpsR={_EpsR}')

        _InsLayer = fields.get('InsLayer')
        if _InsLayer is not None:
            output.write(f' InsLayer={_InsLayer}')

        _DiaIns = fields.get('DiaIns')
        if _DiaIns is not None:
            output.write(f' DiaIns={_DiaIns}')

        _DiaCable = fields.get('DiaCable')
        if _DiaCable is not None:
            output.write(f' DiaCable={_DiaCable}')

        _RDC = fields.get('RDC')
        if _RDC is not None:
            output.write(f' RDC={_RDC}')

        _RAC = fields.get('RAC')
        if _RAC is not None:
            output.write(f' RAC={_RAC}')

        _RUnits = fields.get('RUnits')
        if _RUnits is not None:
            output.write(f' RUnits={_quoted(_RUnits)}')

        _GMRAC = fields.get('GMRAC')
        if _GMRAC is not None:
            output.write(f' GMRAC={_GMRAC}')

        _GMRUnits = fields.get('GMRUnits')
        if _GMRUnits is not None:
            output.write(f' GMRUnits={_quoted(_GMRUnits)}')

        _Radius = fields.get('Radius')
        if _Radius is not None:
            output.write(f' Radius={_Radius}')

        _RadUnits = fields.get('RadUnits')
        if _RadUnits is not None:
            output.write(f' RadUnits={_quoted(_RadUnits)}')

        _NormAmps = fields.get('NormAmps')
        if _NormAmps is not None:
            output.write(f' NormAmps={_NormAmps}')

        _EmergAmps = fields.get('EmergAmps')
        if _EmergAmps is not None:
            output.write(f' EmergAmps={_EmergAmps}')

        _Ratings = fields.get('Ratings')
        if _Ratings is not None:
            if isinstance(_Ratings, ARRAY_LIKE):
                output.write(f' Seasons={len(_Ratings)}')
                _Ratings = _as_list(_Ratings)
            else:
                _length_Ratings, _Ratings = _filepath_array(_Ratings)
                output.write(f' Seasons={_length_Ratings}')
            output.write(f' Ratings={_Ratings}')

        _CapRadius = fields.get('CapRadius')
        if _CapRadius is not None:
            output.write(f' CapRadius={_CapRadius}')

        output.write('\n')


TSData_ = TSData


class TSDataList(RootModel[List[TSData]]):
    root: List[TSData]





class TSDataContainer(RootModel[Union[TSDataList, JSONFilePath, JSONLinesFilePath]]):
    root: Union[TSDataList, JSONFilePath, JSONLinesFilePath] = Field(..., title="TSDataContainer")


    @model_validator(mode="before")
    def _val_dss_model(self) -> "TSDataContainer":
        try:
            _fields_set = set(self.keys())
        except:
            if isinstance(self, (list, tuple)):
                return self

            raise

        # Validate oneOf (spec. sets) based on the `required` lists
        _required_TSDataList = "root" in _fields_set and isinstance(self.root, TSDataList)
        _required_JSONFilePath = _fields_set.issuperset({'JSONFile'})
        _required_JSONLinesFilePath = _fields_set.issuperset({'JSONLinesFile'})
        num_specs = _required_TSDataList + _required_JSONFilePath + _required_JSONLinesFilePath
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



class LineSpacing(BaseModel):
    Name: Annotated[str, Field(max_length=255, min_length=1)] = Field(..., title="Name")
    Like: Optional[str] = Field(None, title="Like")
    NPhases: Optional[int] = Field(None, title="NPhases")
    X: Optional[List[float]] = Field(None, title="X")
    H: Optional[List[float]] = Field(None, title="H")
    Units: Optional[LengthUnit] = Field(None, title="Units")

    def dumps_dss(self, edit=False) -> str:
        """
        Dump the DSS object dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            self.dump_dss(output, edit)
            return output.getvalue()

    def dump_dss(self, output: TextIO, edit=False):
        """
        Dump the DSS object to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        LineSpacing.dict_dump_dss(self.model_dump(exclude_unset=True), output, edit=edit)

    @staticmethod
    def dict_dumps_dss(fields: Dict, edit=False) -> str:
        """
        Dump the DSS object from a dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            LineSpacing.dict_dump_dss(fields, output, edit)
            return output.getvalue()

    @staticmethod
    def dict_dump_dss(fields: Dict, output: TextIO, edit=False):
        """
        Dump the DSS object from a dict to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        _length_NConds = None

        if edit:
            output.write(f'''edit LineSpacing.{fields['Name']}''')
        else:
            output.write(f'''new LineSpacing.{fields['Name']}''')

        _Like = fields.get('Like')
        if _Like is not None:
            output.write(f' Like={_quoted(_Like)}')

        _NPhases = fields.get('NPhases')
        if _NPhases is not None:
            output.write(f' NPhases={_NPhases}')

        _X = fields.get('X')
        if _X is not None:
            _length_NConds = len(_X)
            output.write(f' NConds={_length_NConds}')
            output.write(f' X={_as_list(_X)}')

        _H = fields.get('H')
        if _H is not None:
            if _length_NConds is None:
                _length_NConds = len(_H)
                output.write(f' NConds={_length_NConds}')
            elif len(_H) != _length_NConds:
                raise ValueError(f'Array length ({len(_H)}) for "H" does not match expected length ({_length_NConds})')

            output.write(f' H={_as_list(_H)}')

        _Units = fields.get('Units')
        if _Units is not None:
            output.write(f' Units={_quoted(_Units)}')

        output.write('\n')


LineSpacing_ = LineSpacing


class LineSpacingList(RootModel[List[LineSpacing]]):
    root: List[LineSpacing]





class LineSpacingContainer(RootModel[Union[LineSpacingList, JSONFilePath, JSONLinesFilePath]]):
    root: Union[LineSpacingList, JSONFilePath, JSONLinesFilePath] = Field(..., title="LineSpacingContainer")


    @model_validator(mode="before")
    def _val_dss_model(self) -> "LineSpacingContainer":
        try:
            _fields_set = set(self.keys())
        except:
            if isinstance(self, (list, tuple)):
                return self

            raise

        # Validate oneOf (spec. sets) based on the `required` lists
        _required_LineSpacingList = "root" in _fields_set and isinstance(self.root, LineSpacingList)
        _required_JSONFilePath = _fields_set.issuperset({'JSONFile'})
        _required_JSONLinesFilePath = _fields_set.issuperset({'JSONLinesFile'})
        num_specs = _required_LineSpacingList + _required_JSONFilePath + _required_JSONLinesFilePath
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



class LineGeometry_Common(BaseModel):
    Name: Annotated[str, Field(max_length=255, min_length=1)] = Field(..., title="Name")
    Like: Optional[str] = Field(None, title="Like")
    NConds: Optional[PositiveInt] = Field(None, title="NConds")
    NPhases: Optional[Annotated[int, Field(ge=0)]] = Field(None, title="NPhases")
    Units: Optional[List[LengthUnit]] = Field(None, title="Units")
    NormAmps: Optional[float] = Field(None, title="NormAmps")
    EmergAmps: Optional[float] = Field(None, title="EmergAmps")
    Reduce: Optional[bool] = Field(None, title="Reduce")
    Conductors: Optional[List[str]] = Field(None, title="Wires", validation_alias=AliasChoices("Conductors", "Wires"))
    Ratings: Optional[ArrayOrFilePath] = Field(None, title="Ratings")
    LineType: Optional[LineType_] = Field(None, title="LineType")

class LineGeometry_LineSpacing(LineGeometry_Common):
    Spacing: Annotated[str, Field(max_length=255, min_length=1)] = Field(..., title="Spacing")

class LineGeometry_xh(LineGeometry_Common):
    X: List[float] = Field(..., title="X")
    H: List[float] = Field(..., title="H")


class LineGeometry(RootModel[Union[LineGeometry_LineSpacing, LineGeometry_xh]]):
    root: Union[LineGeometry_LineSpacing, LineGeometry_xh] = Field(..., title="LineGeometry")

    def dumps_dss(self, edit=False) -> str:
        """
        Dump the DSS object dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            self.dump_dss(output, edit)
            return output.getvalue()

    def dump_dss(self, output: TextIO, edit=False):
        """
        Dump the DSS object to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        LineGeometry.dict_dump_dss(self.model_dump(exclude_unset=True), output, edit=edit)

    @staticmethod
    def dict_dumps_dss(fields: Dict, edit=False) -> str:
        """
        Dump the DSS object from a dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            LineGeometry.dict_dump_dss(fields, output, edit)
            return output.getvalue()

    @staticmethod
    def dict_dump_dss(fields: Dict, output: TextIO, edit=False):
        """
        Dump the DSS object from a dict to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        _length_NConds = None

        if edit:
            output.write(f'''edit LineGeometry.{fields['Name']}''')
        else:
            output.write(f'''new LineGeometry.{fields['Name']}''')

        # NOTE: "NConds" is redundant, left for clarity

        _Like = fields.get('Like')
        if _Like is not None:
            output.write(f' Like={_quoted(_Like)}')

        _NConds = fields.get('NConds')
        if _NConds is not None:
            _length_NConds = _NConds
            output.write(f' NConds={_NConds}')

        _NPhases = fields.get('NPhases')
        if _NPhases is not None:
            output.write(f' NPhases={_NPhases}')

        _NormAmps = fields.get('NormAmps')
        if _NormAmps is not None:
            output.write(f' NormAmps={_NormAmps}')

        _EmergAmps = fields.get('EmergAmps')
        if _EmergAmps is not None:
            output.write(f' EmergAmps={_EmergAmps}')

        _Reduce = fields.get('Reduce')
        if _Reduce is not None:
            output.write(f' Reduce={_Reduce}')

        _Spacing = fields.get('Spacing')
        if _Spacing is not None:
            output.write(f' Spacing={_quoted(_Spacing)}')

        _Ratings = fields.get('Ratings')
        if _Ratings is not None:
            if isinstance(_Ratings, ARRAY_LIKE):
                output.write(f' Seasons={len(_Ratings)}')
                _Ratings = _as_list(_Ratings)
            else:
                _length_Ratings, _Ratings = _filepath_array(_Ratings)
                output.write(f' Seasons={_length_Ratings}')
            output.write(f' Ratings={_Ratings}')

        _LineType = fields.get('LineType')
        if _LineType is not None:
            output.write(f' LineType={_quoted(_LineType)}')

        _needs_Cond = False
        _X = fields.get('X')
        if _X is not None:
            if len(_X) != _length_NConds:
                raise ValueError(f'Array length ({len(_X)}) for "X" does not match expected length ({_length_NConds})')
            _needs_Cond = True
        _H = fields.get('H')
        if _H is not None:
            if len(_H) != _length_NConds:
                raise ValueError(f'Array length ({len(_H)}) for "H" does not match expected length ({_length_NConds})')
            _needs_Cond = True
        _Units = fields.get('Units')
        if _Units is not None:
            if len(_Units) != _length_NConds:
                raise ValueError(f'Array length ({len(_Units)}) for "Units" does not match expected length ({_length_NConds})')
            _needs_Cond = True
        _Conductors = fields.get('Conductors')
        if _Conductors is not None:
            if len(_Conductors) != _length_NConds:
                raise ValueError(f'Array length ({len(_Conductors)}) for "Conductors" does not match expected length ({_length_NConds})')
            _needs_Cond = True
        if _length_NConds is not None and _needs_Cond:
            for _Cond in range(_length_NConds):
                output.write(f" Cond={_Cond + 1}")
                _X = fields.get('X')
                if _X is not None:
                    output.write(f" X={_X[_Cond]}")
                _H = fields.get('H')
                if _H is not None:
                    output.write(f" H={_H[_Cond]}")
                _Units = fields.get('Units')
                if _Units is not None:
                    output.write(f" Units={_Units[_Cond]}")
                _Conductors = fields.get("Conductors") or []
                if _Conductors:
                    cnd_cls, cnd_name = _Conductors[_Cond].split('.', 1)
                    cnd_cls = cnd_cls.lower()
                    if cnd_cls == 'wiredata':
                        output.write(f" wire={_quoted(cnd_name)}")
                    elif cnd_cls == 'cndata':
                        output.write(f" cncable={_quoted(cnd_name)}")
                    elif cnd_cls == 'tsdata':
                        output.write(f" tscable={_quoted(cnd_name)}")
                    else:
                        raise ValueError(f'Could not match object type for element in Conductors array: "{_Conductors[_Cond]}".')

        output.write('\n')

    @model_validator(mode="before")
    def _val_dss_model(self) -> "LineGeometry":
        _fields_set = set(self.keys())
        # Validate oneOf (spec. sets) based on the `required` lists
        _required_LineSpacing = _fields_set.issuperset({'Spacing'})
        _required_xh = _fields_set.issuperset({'H', 'X'})
        num_specs = _required_LineSpacing + _required_xh
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



LineGeometry_ = LineGeometry


class LineGeometryList(RootModel[List[LineGeometry]]):
    root: List[LineGeometry]





class LineGeometryContainer(RootModel[Union[LineGeometryList, JSONFilePath, JSONLinesFilePath]]):
    root: Union[LineGeometryList, JSONFilePath, JSONLinesFilePath] = Field(..., title="LineGeometryContainer")


    @model_validator(mode="before")
    def _val_dss_model(self) -> "LineGeometryContainer":
        try:
            _fields_set = set(self.keys())
        except:
            if isinstance(self, (list, tuple)):
                return self

            raise

        # Validate oneOf (spec. sets) based on the `required` lists
        _required_LineGeometryList = "root" in _fields_set and isinstance(self.root, LineGeometryList)
        _required_JSONFilePath = _fields_set.issuperset({'JSONFile'})
        _required_JSONLinesFilePath = _fields_set.issuperset({'JSONLinesFile'})
        num_specs = _required_LineGeometryList + _required_JSONFilePath + _required_JSONLinesFilePath
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



class XfmrCode_Common(BaseModel):
    Name: Annotated[str, Field(max_length=255, min_length=1)] = Field(..., title="Name")
    Like: Optional[str] = Field(None, title="Like")
    Phases: Optional[PositiveInt] = Field(None, title="Phases")
    Conn: Optional[List[Connection]] = Field(None, title="Conn")
    kV: List[float] = Field(..., title="kV")
    kVA: Optional[List[float]] = Field(None, title="kVA")
    Tap: Optional[List[float]] = Field(None, title="Tap")
    pctR: Optional[List[float]] = Field(None, title="%R", validation_alias=AliasChoices("pctR", "%R"))
    RNeut: Optional[List[float]] = Field(None, title="RNeut")
    XNeut: Optional[List[float]] = Field(None, title="XNeut")
    Thermal: Optional[float] = Field(None, title="Thermal")
    n: Optional[float] = Field(None, title="n")
    m: Optional[float] = Field(None, title="m")
    FLRise: Optional[float] = Field(None, title="FLRise")
    HSRise: Optional[float] = Field(None, title="HSRise")
    pctLoadLoss: Optional[float] = Field(None, title="%LoadLoss", validation_alias=AliasChoices("pctLoadLoss", "%LoadLoss"))
    pctNoLoadLoss: Optional[float] = Field(None, title="%NoLoadLoss", validation_alias=AliasChoices("pctNoLoadLoss", "%NoLoadLoss"))
    NormHkVA: Optional[float] = Field(None, title="NormHkVA")
    EmergHkVA: Optional[float] = Field(None, title="EmergHkVA")
    MaxTap: Optional[List[float]] = Field(None, title="MaxTap")
    MinTap: Optional[List[float]] = Field(None, title="MinTap")
    NumTaps: Optional[List[int]] = Field(None, title="NumTaps")
    pctIMag: Optional[float] = Field(None, title="%IMag", validation_alias=AliasChoices("pctIMag", "%IMag"))
    ppm_Antifloat: Optional[float] = Field(None, title="ppm_Antifloat")
    RDCOhms: Optional[List[float]] = Field(None, title="RDCOhms")
    Ratings: Optional[ArrayOrFilePath] = Field(None, title="Ratings")

class XfmrCode_X12X13X23(XfmrCode_Common):
    X12: float = Field(..., title="X12")
    X13: Optional[float] = Field(None, title="X13")
    X23: Optional[float] = Field(None, title="X23")

class XfmrCode_XscArray(XfmrCode_Common):
    XSCArray: List[float] = Field(..., title="XSCArray")


class XfmrCode(RootModel[Union[XfmrCode_X12X13X23, XfmrCode_XscArray]]):
    root: Union[XfmrCode_X12X13X23, XfmrCode_XscArray] = Field(..., title="XfmrCode")

    def dumps_dss(self, edit=False) -> str:
        """
        Dump the DSS object dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            self.dump_dss(output, edit)
            return output.getvalue()

    def dump_dss(self, output: TextIO, edit=False):
        """
        Dump the DSS object to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        XfmrCode.dict_dump_dss(self.model_dump(exclude_unset=True), output, edit=edit)

    @staticmethod
    def dict_dumps_dss(fields: Dict, edit=False) -> str:
        """
        Dump the DSS object from a dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            XfmrCode.dict_dump_dss(fields, output, edit)
            return output.getvalue()

    @staticmethod
    def dict_dump_dss(fields: Dict, output: TextIO, edit=False):
        """
        Dump the DSS object from a dict to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        _length_Windings = None

        if edit:
            output.write(f'''edit XfmrCode.{fields['Name']}''')
        else:
            output.write(f'''new XfmrCode.{fields['Name']}''')

        _Like = fields.get('Like')
        if _Like is not None:
            output.write(f' Like={_quoted(_Like)}')

        _Phases = fields.get('Phases')
        if _Phases is not None:
            output.write(f' Phases={_Phases}')

        _Conn = fields.get('Conn')
        if _Conn is not None:
            _length_Windings = len(_Conn)
            output.write(f' Windings={_length_Windings}')
            output.write(f' Conns=({_quoted_list(_Conn)})')

        _kV = fields.get('kV')
        if _length_Windings is None:
            _length_Windings = len(_kV)
            output.write(f' Windings={_length_Windings}')
        elif len(_kV) != _length_Windings:
            raise ValueError(f'Array length ({len(_kV)}) for "kV" does not match expected length ({_length_Windings})')

        output.write(f' kVs={_as_list(_kV)}')
        _kVA = fields.get('kVA')
        if _kVA is not None:
            if _length_Windings is None:
                _length_Windings = len(_kVA)
                output.write(f' Windings={_length_Windings}')
            elif len(_kVA) != _length_Windings:
                raise ValueError(f'Array length ({len(_kVA)}) for "kVA" does not match expected length ({_length_Windings})')

            output.write(f' kVAs={_as_list(_kVA)}')

        _Tap = fields.get('Tap')
        if _Tap is not None:
            if _length_Windings is None:
                _length_Windings = len(_Tap)
                output.write(f' Windings={_length_Windings}')
            elif len(_Tap) != _length_Windings:
                raise ValueError(f'Array length ({len(_Tap)}) for "Tap" does not match expected length ({_length_Windings})')

            output.write(f' Taps={_as_list(_Tap)}')

        _pctR = fields.get('pctR')
        if _pctR is not None:
            if _length_Windings is None:
                _length_Windings = len(_pctR)
                output.write(f' Windings={_length_Windings}')
            elif len(_pctR) != _length_Windings:
                raise ValueError(f'Array length ({len(_pctR)}) for "pctR" does not match expected length ({_length_Windings})')

            output.write(f' %Rs={_as_list(_pctR)}')

        _XSCArray = fields.get('XSCArray')
        if _XSCArray is not None:
            output.write(f' XSCArray={_as_list(_XSCArray)}')

        _Thermal = fields.get('Thermal')
        if _Thermal is not None:
            output.write(f' Thermal={_Thermal}')

        _n = fields.get('n')
        if _n is not None:
            output.write(f' n={_n}')

        _m = fields.get('m')
        if _m is not None:
            output.write(f' m={_m}')

        _FLRise = fields.get('FLRise')
        if _FLRise is not None:
            output.write(f' FLRise={_FLRise}')

        _HSRise = fields.get('HSRise')
        if _HSRise is not None:
            output.write(f' HSRise={_HSRise}')

        _pctLoadLoss = fields.get('pctLoadLoss')
        if _pctLoadLoss is not None:
            output.write(f' %LoadLoss={_pctLoadLoss}')

        _pctNoLoadLoss = fields.get('pctNoLoadLoss')
        if _pctNoLoadLoss is not None:
            output.write(f' %NoLoadLoss={_pctNoLoadLoss}')

        _NormHkVA = fields.get('NormHkVA')
        if _NormHkVA is not None:
            output.write(f' NormHkVA={_NormHkVA}')

        _EmergHkVA = fields.get('EmergHkVA')
        if _EmergHkVA is not None:
            output.write(f' EmergHkVA={_EmergHkVA}')

        _pctIMag = fields.get('pctIMag')
        if _pctIMag is not None:
            output.write(f' %IMag={_pctIMag}')

        _ppm_Antifloat = fields.get('ppm_Antifloat')
        if _ppm_Antifloat is not None:
            output.write(f' ppm_Antifloat={_ppm_Antifloat}')

        _X12 = fields.get('X12')
        if _X12 is not None:
            output.write(f' X12={_X12}')

        _X13 = fields.get('X13')
        if _X13 is not None:
            output.write(f' X13={_X13}')

        _X23 = fields.get('X23')
        if _X23 is not None:
            output.write(f' X23={_X23}')

        _Ratings = fields.get('Ratings')
        if _Ratings is not None:
            if isinstance(_Ratings, ARRAY_LIKE):
                output.write(f' Seasons={len(_Ratings)}')
                _Ratings = _as_list(_Ratings)
            else:
                _length_Ratings, _Ratings = _filepath_array(_Ratings)
                output.write(f' Seasons={_length_Ratings}')
            output.write(f' Ratings={_Ratings}')

        _needs_Wdg = False
        _RNeut = fields.get('RNeut')
        if _RNeut is not None:
            if len(_RNeut) != _length_Windings:
                raise ValueError(f'Array length ({len(_RNeut)}) for "RNeut" does not match expected length ({_length_Windings})')
            _needs_Wdg = True
        _XNeut = fields.get('XNeut')
        if _XNeut is not None:
            if len(_XNeut) != _length_Windings:
                raise ValueError(f'Array length ({len(_XNeut)}) for "XNeut" does not match expected length ({_length_Windings})')
            _needs_Wdg = True
        _MaxTap = fields.get('MaxTap')
        if _MaxTap is not None:
            if len(_MaxTap) != _length_Windings:
                raise ValueError(f'Array length ({len(_MaxTap)}) for "MaxTap" does not match expected length ({_length_Windings})')
            _needs_Wdg = True
        _MinTap = fields.get('MinTap')
        if _MinTap is not None:
            if len(_MinTap) != _length_Windings:
                raise ValueError(f'Array length ({len(_MinTap)}) for "MinTap" does not match expected length ({_length_Windings})')
            _needs_Wdg = True
        _NumTaps = fields.get('NumTaps')
        if _NumTaps is not None:
            if len(_NumTaps) != _length_Windings:
                raise ValueError(f'Array length ({len(_NumTaps)}) for "NumTaps" does not match expected length ({_length_Windings})')
            _needs_Wdg = True
        _RDCOhms = fields.get('RDCOhms')
        if _RDCOhms is not None:
            if len(_RDCOhms) != _length_Windings:
                raise ValueError(f'Array length ({len(_RDCOhms)}) for "RDCOhms" does not match expected length ({_length_Windings})')
            _needs_Wdg = True
        if _length_Windings is not None and _needs_Wdg:
            for _Wdg in range(_length_Windings):
                output.write(f" Wdg={_Wdg + 1}")
                _RNeut = fields.get('RNeut')
                if _RNeut is not None:
                    output.write(f" RNeut={_RNeut[_Wdg]}")
                _XNeut = fields.get('XNeut')
                if _XNeut is not None:
                    output.write(f" XNeut={_XNeut[_Wdg]}")
                _MaxTap = fields.get('MaxTap')
                if _MaxTap is not None:
                    output.write(f" MaxTap={_MaxTap[_Wdg]}")
                _MinTap = fields.get('MinTap')
                if _MinTap is not None:
                    output.write(f" MinTap={_MinTap[_Wdg]}")
                _NumTaps = fields.get('NumTaps')
                if _NumTaps is not None:
                    output.write(f" NumTaps={_NumTaps[_Wdg]}")
                _RDCOhms = fields.get('RDCOhms')
                if _RDCOhms is not None:
                    output.write(f" RDCOhms={_RDCOhms[_Wdg]}")
        output.write('\n')

    @model_validator(mode="before")
    def _val_dss_model(self) -> "XfmrCode":
        _fields_set = set(self.keys())
        # Validate oneOf (spec. sets) based on the `required` lists
        _required_X12X13X23 = _fields_set.issuperset({'X12'})
        _required_XscArray = _fields_set.issuperset({'XSCArray'})
        num_specs = _required_X12X13X23 + _required_XscArray
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



XfmrCode_ = XfmrCode


class XfmrCodeList(RootModel[List[XfmrCode]]):
    root: List[XfmrCode]





class XfmrCodeContainer(RootModel[Union[XfmrCodeList, JSONFilePath, JSONLinesFilePath]]):
    root: Union[XfmrCodeList, JSONFilePath, JSONLinesFilePath] = Field(..., title="XfmrCodeContainer")


    @model_validator(mode="before")
    def _val_dss_model(self) -> "XfmrCodeContainer":
        try:
            _fields_set = set(self.keys())
        except:
            if isinstance(self, (list, tuple)):
                return self

            raise

        # Validate oneOf (spec. sets) based on the `required` lists
        _required_XfmrCodeList = "root" in _fields_set and isinstance(self.root, XfmrCodeList)
        _required_JSONFilePath = _fields_set.issuperset({'JSONFile'})
        _required_JSONLinesFilePath = _fields_set.issuperset({'JSONLinesFile'})
        num_specs = _required_XfmrCodeList + _required_JSONFilePath + _required_JSONLinesFilePath
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



class Line_Common(BaseModel):
    Name: Annotated[str, Field(max_length=255, min_length=1)] = Field(..., title="Name")
    Like: Optional[str] = Field(None, title="Like")
    Switch: Optional[bool] = Field(None, title="Switch")
    Bus1: BusConnection = Field(..., title="Bus1")
    Bus2: BusConnection = Field(..., title="Bus2")
    Phases: Optional[PositiveInt] = Field(None, title="Phases")
    Rg: Optional[float] = Field(None, title="Rg")
    Xg: Optional[float] = Field(None, title="Xg")
    rho: Optional[float] = Field(None, title="rho")
    Units: Optional[LengthUnit] = Field(None, title="Units")
    EarthModel: Optional[EarthModel_] = Field(None, title="EarthModel")
    Ratings: Optional[ArrayOrFilePath] = Field(None, title="Ratings")
    LineType: Optional[LineType_] = Field(None, title="LineType")
    NormAmps: Optional[float] = Field(None, title="NormAmps")
    EmergAmps: Optional[float] = Field(None, title="EmergAmps")
    FaultRate: Optional[float] = Field(None, title="FaultRate")
    pctPerm: Optional[float] = Field(None, title="pctPerm")
    Repair: Optional[float] = Field(None, title="Repair")
    BaseFreq: Optional[PositiveFloat] = Field(None, title="BaseFreq")
    Enabled: Optional[bool] = Field(None, title="Enabled")

class Line_LineCode(Line_Common):
    LineCode: Annotated[str, Field(max_length=255, min_length=1)] = Field(..., title="LineCode")
    Length: Optional[float] = Field(None, title="Length")

class Line_LineGeometry(Line_Common):
    Length: Optional[float] = Field(None, title="Length")
    Geometry: Annotated[str, Field(max_length=255, min_length=1)] = Field(..., title="Geometry")

class Line_SpacingWires(Line_Common):
    Length: Optional[float] = Field(None, title="Length")
    Spacing: Annotated[str, Field(max_length=255, min_length=1)] = Field(..., title="Spacing")
    Conductors: List[str] = Field(..., title="Wires", validation_alias=AliasChoices("Conductors", "Wires"))

class Line_Z0Z1C0C1(Line_Common):
    R1: float = Field(..., title="R1")
    X1: float = Field(..., title="X1")
    R0: Optional[float] = Field(None, title="R0")
    X0: Optional[float] = Field(None, title="X0")
    C1: float = Field(..., title="C1")
    C0: Optional[float] = Field(None, title="C0")

class Line_ZMatrixCMatrix(Line_Common):
    RMatrix: SymmetricMatrix = Field(..., title="RMatrix")
    XMatrix: SymmetricMatrix = Field(..., title="XMatrix")
    CMatrix: Optional[SymmetricMatrix] = Field(None, title="CMatrix")

    @field_validator('RMatrix')
    @classmethod
    def _RMatrix_check_symm_matrix(cls, value: Optional[SymmetricMatrix], info: ValidationInfo) -> Optional[SymmetricMatrix]:
        return _check_symmetric_matrix(value, info.data.get('Phases'))

    @field_validator('XMatrix')
    @classmethod
    def _XMatrix_check_symm_matrix(cls, value: Optional[SymmetricMatrix], info: ValidationInfo) -> Optional[SymmetricMatrix]:
        return _check_symmetric_matrix(value, info.data.get('Phases'))

    @field_validator('CMatrix')
    @classmethod
    def _CMatrix_check_symm_matrix(cls, value: Optional[SymmetricMatrix], info: ValidationInfo) -> Optional[SymmetricMatrix]:
        return _check_symmetric_matrix(value, info.data.get('Phases'))


class Line(RootModel[Union[Line_LineCode, Line_LineGeometry, Line_SpacingWires, Line_Z0Z1C0C1, Line_ZMatrixCMatrix]]):
    root: Union[Line_LineCode, Line_LineGeometry, Line_SpacingWires, Line_Z0Z1C0C1, Line_ZMatrixCMatrix] = Field(..., title="Line")

    def dumps_dss(self, edit=False) -> str:
        """
        Dump the DSS object dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            self.dump_dss(output, edit)
            return output.getvalue()

    def dump_dss(self, output: TextIO, edit=False):
        """
        Dump the DSS object to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        Line.dict_dump_dss(self.model_dump(exclude_unset=True), output, edit=edit)

    @staticmethod
    def dict_dumps_dss(fields: Dict, edit=False) -> str:
        """
        Dump the DSS object from a dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            Line.dict_dump_dss(fields, output, edit)
            return output.getvalue()

    @staticmethod
    def dict_dump_dss(fields: Dict, output: TextIO, edit=False):
        """
        Dump the DSS object from a dict to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        if edit:
            output.write(f'''edit Line.{fields['Name']}''')
        else:
            output.write(f'''new Line.{fields['Name']}''')

        _Like = fields.get('Like')
        if _Like is not None:
            output.write(f' Like={_quoted(_Like)}')

        _Switch = fields.get('Switch')
        if _Switch is not None:
            output.write(f' Switch={_Switch}')

        _Bus1 = fields.get('Bus1')
        output.write(f' Bus1={_quoted(_Bus1)}')
        _Bus2 = fields.get('Bus2')
        output.write(f' Bus2={_quoted(_Bus2)}')
        _LineCode = fields.get('LineCode')
        if _LineCode is not None:
            output.write(f' LineCode={_quoted(_LineCode)}')

        _Length = fields.get('Length')
        if _Length is not None:
            output.write(f' Length={_Length}')

        _Phases = fields.get('Phases')
        if _Phases is not None:
            output.write(f' Phases={_Phases}')

        _R1 = fields.get('R1')
        if _R1 is not None:
            output.write(f' R1={_R1}')

        _X1 = fields.get('X1')
        if _X1 is not None:
            output.write(f' X1={_X1}')

        _R0 = fields.get('R0')
        if _R0 is not None:
            output.write(f' R0={_R0}')

        _X0 = fields.get('X0')
        if _X0 is not None:
            output.write(f' X0={_X0}')

        _C1 = fields.get('C1')
        if _C1 is not None:
            output.write(f' C1={_C1}')

        _C0 = fields.get('C0')
        if _C0 is not None:
            output.write(f' C0={_C0}')

        _RMatrix = fields.get('RMatrix')
        if _RMatrix is not None:
            output.write(_dump_symmetric_matrix("RMatrix", _RMatrix))

        _XMatrix = fields.get('XMatrix')
        if _XMatrix is not None:
            output.write(_dump_symmetric_matrix("XMatrix", _XMatrix))

        _CMatrix = fields.get('CMatrix')
        if _CMatrix is not None:
            output.write(_dump_symmetric_matrix("CMatrix", _CMatrix))

        _Rg = fields.get('Rg')
        if _Rg is not None:
            output.write(f' Rg={_Rg}')

        _Xg = fields.get('Xg')
        if _Xg is not None:
            output.write(f' Xg={_Xg}')

        _rho = fields.get('rho')
        if _rho is not None:
            output.write(f' rho={_rho}')

        _Geometry = fields.get('Geometry')
        if _Geometry is not None:
            output.write(f' Geometry={_quoted(_Geometry)}')

        _Units = fields.get('Units')
        if _Units is not None:
            output.write(f' Units={_quoted(_Units)}')

        _Spacing = fields.get('Spacing')
        if _Spacing is not None:
            output.write(f' Spacing={_quoted(_Spacing)}')

        _EarthModel = fields.get('EarthModel')
        if _EarthModel is not None:
            output.write(f' EarthModel={_quoted(_EarthModel)}')

        _Ratings = fields.get('Ratings')
        if _Ratings is not None:
            if isinstance(_Ratings, ARRAY_LIKE):
                output.write(f' Seasons={len(_Ratings)}')
                _Ratings = _as_list(_Ratings)
            else:
                _length_Ratings, _Ratings = _filepath_array(_Ratings)
                output.write(f' Seasons={_length_Ratings}')
            output.write(f' Ratings={_Ratings}')

        _LineType = fields.get('LineType')
        if _LineType is not None:
            output.write(f' LineType={_quoted(_LineType)}')

        _NormAmps = fields.get('NormAmps')
        if _NormAmps is not None:
            output.write(f' NormAmps={_NormAmps}')

        _EmergAmps = fields.get('EmergAmps')
        if _EmergAmps is not None:
            output.write(f' EmergAmps={_EmergAmps}')

        _FaultRate = fields.get('FaultRate')
        if _FaultRate is not None:
            output.write(f' FaultRate={_FaultRate}')

        _pctPerm = fields.get('pctPerm')
        if _pctPerm is not None:
            output.write(f' pctPerm={_pctPerm}')

        _Repair = fields.get('Repair')
        if _Repair is not None:
            output.write(f' Repair={_Repair}')

        _BaseFreq = fields.get('BaseFreq')
        if _BaseFreq is not None:
            output.write(f' BaseFreq={_BaseFreq}')

        _Enabled = fields.get('Enabled')
        if _Enabled is not None:
            output.write(f' Enabled={_Enabled}')

        _Conductors = fields.get("Conductors") or []
        if len(_Conductors) > 0:
            _cnds = list(_Conductors)
            _done = defaultdict(list)
            _prev_cnd_cls = None
            while _cnds:
                cnd = _cnds.pop(0).lower()
                cnd_cls, cnd_name = cnd.split('.', 1)
                cnd_cls = cnd_cls.lower()
                if _prev_cnd_cls != cnd_cls and cnd_cls in _done:
                    raise ValueError(f'Conductor types are not contiguous.')

                if cnd_cls in ('wiredata', 'cndata', 'tsdata'):
                    _done[cnd_cls].append(cnd_name)
                    _prev_cnd_cls = cnd_cls
                else:
                    raise ValueError(f'Could not match object type for element in Conductors array: "{cnd}".')
            for cnd_cls, cnd_objs in _done.items():
                if cnd_cls == 'wiredata':
                    output.write(f" wires=({_quoted_list(cnd_objs)})")
                elif cnd_cls == 'cndata':
                    output.write(f" cncables=({_quoted_list(cnd_objs)})")
                elif cnd_cls == 'tsdata':
                    output.write(f" tscables=({_quoted_list(cnd_objs)})")

        output.write('\n')

    @model_validator(mode="before")
    def _val_dss_model(self) -> "Line":
        _fields_set = set(self.keys())
        # Validate oneOf (spec. sets) based on the `required` lists
        _required_LineCode = _fields_set.issuperset({'LineCode'})
        _required_LineGeometry = _fields_set.issuperset({'Geometry'})
        _required_SpacingWires = _fields_set.issuperset({'Conductors', 'Spacing'})
        _required_Z0Z1C0C1 = _fields_set.issuperset({'C1', 'R1', 'X1'})
        _required_ZMatrixCMatrix = _fields_set.issuperset({'RMatrix', 'XMatrix'})
        num_specs = _required_LineCode + _required_LineGeometry + _required_SpacingWires + _required_Z0Z1C0C1 + _required_ZMatrixCMatrix
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



Line_ = Line


class LineList(RootModel[List[Line]]):
    root: List[Line]





class LineContainer(RootModel[Union[LineList, JSONFilePath, JSONLinesFilePath]]):
    root: Union[LineList, JSONFilePath, JSONLinesFilePath] = Field(..., title="LineContainer")


    @model_validator(mode="before")
    def _val_dss_model(self) -> "LineContainer":
        try:
            _fields_set = set(self.keys())
        except:
            if isinstance(self, (list, tuple)):
                return self

            raise

        # Validate oneOf (spec. sets) based on the `required` lists
        _required_LineList = "root" in _fields_set and isinstance(self.root, LineList)
        _required_JSONFilePath = _fields_set.issuperset({'JSONFile'})
        _required_JSONLinesFilePath = _fields_set.issuperset({'JSONLinesFile'})
        num_specs = _required_LineList + _required_JSONFilePath + _required_JSONLinesFilePath
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



class Vsource_Common(BaseModel):
    Name: Annotated[str, Field(max_length=255, min_length=1)] = Field(..., title="Name")
    Like: Optional[str] = Field(None, title="Like")
    Bus1: BusConnection = Field(..., title="Bus1")
    BasekV: float = Field(..., title="BasekV")
    pu: Optional[float] = Field(None, title="pu")
    Angle: Optional[float] = Field(None, title="Angle")
    Frequency: Optional[PositiveFloat] = Field(None, title="Frequency")
    Phases: Optional[PositiveInt] = Field(None, title="Phases")
    ScanType: Optional[ScanType_] = Field(None, title="ScanType")
    Sequence: Optional[SequenceType] = Field(None, title="Sequence")
    Bus2: Optional[BusConnection] = Field(None, title="Bus2")
    Yearly: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="Yearly")
    Daily: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="Daily")
    Duty: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="Duty")
    Model: Optional[VSourceModel] = Field(None, title="Model")
    puZIdeal: Optional[Complex] = Field(None, title="puZIdeal")
    Spectrum: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="Spectrum")
    BaseFreq: Optional[PositiveFloat] = Field(None, title="BaseFreq")
    Enabled: Optional[bool] = Field(None, title="Enabled")

class Vsource_MVAsc3MVAsc1x1r1x0r0(Vsource_Common):
    MVASC3: float = Field(..., title="MVASC3")
    MVASC1: Optional[float] = Field(None, title="MVASC1")
    X1R1: Optional[float] = Field(None, title="X1R1")
    X0R0: Optional[float] = Field(None, title="X0R0")

class Vsource_Isc3Isc1x1r1x0r0(Vsource_Common):
    X1R1: Optional[float] = Field(None, title="X1R1")
    X0R0: Optional[float] = Field(None, title="X0R0")
    Isc3: float = Field(..., title="Isc3")
    Isc1: Optional[float] = Field(None, title="Isc1")

class Vsource_BaseMVApuZ0puZ1puZ2(Vsource_Common):
    puZ1: Complex = Field(..., title="puZ1")
    puZ0: Optional[Complex] = Field(None, title="puZ0")
    puZ2: Optional[Complex] = Field(None, title="puZ2")
    BaseMVA: float = Field(..., title="BaseMVA")

class Vsource_Z0Z1Z2(Vsource_Common):
    Z1: Complex = Field(..., title="Z1")
    Z0: Optional[Complex] = Field(None, title="Z0")
    Z2: Optional[Complex] = Field(None, title="Z2")


class Vsource(RootModel[Union[Vsource_MVAsc3MVAsc1x1r1x0r0, Vsource_Isc3Isc1x1r1x0r0, Vsource_BaseMVApuZ0puZ1puZ2, Vsource_Z0Z1Z2]]):
    root: Union[Vsource_MVAsc3MVAsc1x1r1x0r0, Vsource_Isc3Isc1x1r1x0r0, Vsource_BaseMVApuZ0puZ1puZ2, Vsource_Z0Z1Z2] = Field(..., title="Vsource")

    def dumps_dss(self, edit=False) -> str:
        """
        Dump the DSS object dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            self.dump_dss(output, edit)
            return output.getvalue()

    def dump_dss(self, output: TextIO, edit=False):
        """
        Dump the DSS object to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        Vsource.dict_dump_dss(self.model_dump(exclude_unset=True), output, edit=edit)

    @staticmethod
    def dict_dumps_dss(fields: Dict, edit=False) -> str:
        """
        Dump the DSS object from a dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            Vsource.dict_dump_dss(fields, output, edit)
            return output.getvalue()

    @staticmethod
    def dict_dump_dss(fields: Dict, output: TextIO, edit=False):
        """
        Dump the DSS object from a dict to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        if edit:
            output.write(f'''edit Vsource.{fields['Name']}''')
        else:
            output.write(f'''new Vsource.{fields['Name']}''')

        _Like = fields.get('Like')
        if _Like is not None:
            output.write(f' Like={_quoted(_Like)}')

        _Bus1 = fields.get('Bus1')
        output.write(f' Bus1={_quoted(_Bus1)}')
        _BasekV = fields.get('BasekV')
        output.write(f' BasekV={_BasekV}')
        _pu = fields.get('pu')
        if _pu is not None:
            output.write(f' pu={_pu}')

        _Angle = fields.get('Angle')
        if _Angle is not None:
            output.write(f' Angle={_Angle}')

        _Frequency = fields.get('Frequency')
        if _Frequency is not None:
            output.write(f' Frequency={_Frequency}')

        _Phases = fields.get('Phases')
        if _Phases is not None:
            output.write(f' Phases={_Phases}')

        _MVASC3 = fields.get('MVASC3')
        if _MVASC3 is not None:
            output.write(f' MVASC3={_MVASC3}')

        _MVASC1 = fields.get('MVASC1')
        if _MVASC1 is not None:
            output.write(f' MVASC1={_MVASC1}')

        _X1R1 = fields.get('X1R1')
        if _X1R1 is not None:
            output.write(f' X1R1={_X1R1}')

        _X0R0 = fields.get('X0R0')
        if _X0R0 is not None:
            output.write(f' X0R0={_X0R0}')

        _Isc3 = fields.get('Isc3')
        if _Isc3 is not None:
            output.write(f' Isc3={_Isc3}')

        _Isc1 = fields.get('Isc1')
        if _Isc1 is not None:
            output.write(f' Isc1={_Isc1}')

        _ScanType = fields.get('ScanType')
        if _ScanType is not None:
            output.write(f' ScanType={_quoted(_ScanType)}')

        _Sequence = fields.get('Sequence')
        if _Sequence is not None:
            output.write(f' Sequence={_quoted(_Sequence)}')

        _Bus2 = fields.get('Bus2')
        if _Bus2 is not None:
            output.write(f' Bus2={_quoted(_Bus2)}')

        _Z1 = fields.get('Z1')
        if _Z1 is not None:
            output.write(f' Z1={_complex_to_list(_Z1)}')

        _Z0 = fields.get('Z0')
        if _Z0 is not None:
            output.write(f' Z0={_complex_to_list(_Z0)}')

        _Z2 = fields.get('Z2')
        if _Z2 is not None:
            output.write(f' Z2={_complex_to_list(_Z2)}')

        _puZ1 = fields.get('puZ1')
        if _puZ1 is not None:
            output.write(f' puZ1={_complex_to_list(_puZ1)}')

        _puZ0 = fields.get('puZ0')
        if _puZ0 is not None:
            output.write(f' puZ0={_complex_to_list(_puZ0)}')

        _puZ2 = fields.get('puZ2')
        if _puZ2 is not None:
            output.write(f' puZ2={_complex_to_list(_puZ2)}')

        _BaseMVA = fields.get('BaseMVA')
        if _BaseMVA is not None:
            output.write(f' BaseMVA={_BaseMVA}')

        _Yearly = fields.get('Yearly')
        if _Yearly is not None:
            output.write(f' Yearly={_quoted(_Yearly)}')

        _Daily = fields.get('Daily')
        if _Daily is not None:
            output.write(f' Daily={_quoted(_Daily)}')

        _Duty = fields.get('Duty')
        if _Duty is not None:
            output.write(f' Duty={_quoted(_Duty)}')

        _Model = fields.get('Model')
        if _Model is not None:
            output.write(f' Model={_quoted(_Model)}')

        _puZIdeal = fields.get('puZIdeal')
        if _puZIdeal is not None:
            output.write(f' puZIdeal={_complex_to_list(_puZIdeal)}')

        _Spectrum = fields.get('Spectrum')
        if _Spectrum is not None:
            output.write(f' Spectrum={_quoted(_Spectrum)}')

        _BaseFreq = fields.get('BaseFreq')
        if _BaseFreq is not None:
            output.write(f' BaseFreq={_BaseFreq}')

        _Enabled = fields.get('Enabled')
        if _Enabled is not None:
            output.write(f' Enabled={_Enabled}')

        output.write('\n')

    @model_validator(mode="before")
    def _val_dss_model(self) -> "Vsource":
        _fields_set = set(self.keys())
        # Validate oneOf (spec. sets) based on the `required` lists
        _required_MVAsc3MVAsc1x1r1x0r0 = _fields_set.issuperset({'MVASC3'})
        _required_Isc3Isc1x1r1x0r0 = _fields_set.issuperset({'Isc3'})
        _required_BaseMVApuZ0puZ1puZ2 = _fields_set.issuperset({'BaseMVA', 'puZ1'})
        _required_Z0Z1Z2 = _fields_set.issuperset({'Z1'})
        num_specs = _required_MVAsc3MVAsc1x1r1x0r0 + _required_Isc3Isc1x1r1x0r0 + _required_BaseMVApuZ0puZ1puZ2 + _required_Z0Z1Z2
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



Vsource_ = Vsource


class VsourceList(RootModel[Annotated[List[Vsource], Field(min_length=1)]]):
    root: Annotated[List[Vsource], Field(min_length=1)]





class VsourceContainer(RootModel[Union[VsourceList, JSONFilePath, JSONLinesFilePath]]):
    root: Union[VsourceList, JSONFilePath, JSONLinesFilePath] = Field(..., title="VsourceContainer")


    @model_validator(mode="before")
    def _val_dss_model(self) -> "VsourceContainer":
        try:
            _fields_set = set(self.keys())
        except:
            if isinstance(self, (list, tuple)):
                return self

            raise

        # Validate oneOf (spec. sets) based on the `required` lists
        _required_VsourceList = "root" in _fields_set and isinstance(self.root, VsourceList)
        _required_JSONFilePath = _fields_set.issuperset({'JSONFile'})
        _required_JSONLinesFilePath = _fields_set.issuperset({'JSONLinesFile'})
        num_specs = _required_VsourceList + _required_JSONFilePath + _required_JSONLinesFilePath
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



class Isource(BaseModel):
    Name: Annotated[str, Field(max_length=255, min_length=1)] = Field(..., title="Name")
    Like: Optional[str] = Field(None, title="Like")
    Bus1: BusConnection = Field(..., title="Bus1")
    Amps: Optional[float] = Field(None, title="Amps")
    Angle: Optional[float] = Field(None, title="Angle")
    Frequency: Optional[PositiveFloat] = Field(None, title="Frequency")
    Phases: Optional[PositiveInt] = Field(None, title="Phases")
    ScanType: Optional[ScanType_] = Field(None, title="ScanType")
    Sequence: Optional[SequenceType] = Field(None, title="Sequence")
    Yearly: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="Yearly")
    Daily: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="Daily")
    Duty: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="Duty")
    Bus2: Optional[BusConnection] = Field(None, title="Bus2")
    Spectrum: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="Spectrum")
    BaseFreq: Optional[PositiveFloat] = Field(None, title="BaseFreq")
    Enabled: Optional[bool] = Field(None, title="Enabled")

    def dumps_dss(self, edit=False) -> str:
        """
        Dump the DSS object dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            self.dump_dss(output, edit)
            return output.getvalue()

    def dump_dss(self, output: TextIO, edit=False):
        """
        Dump the DSS object to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        Isource.dict_dump_dss(self.model_dump(exclude_unset=True), output, edit=edit)

    @staticmethod
    def dict_dumps_dss(fields: Dict, edit=False) -> str:
        """
        Dump the DSS object from a dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            Isource.dict_dump_dss(fields, output, edit)
            return output.getvalue()

    @staticmethod
    def dict_dump_dss(fields: Dict, output: TextIO, edit=False):
        """
        Dump the DSS object from a dict to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        if edit:
            output.write(f'''edit Isource.{fields['Name']}''')
        else:
            output.write(f'''new Isource.{fields['Name']}''')

        _Like = fields.get('Like')
        if _Like is not None:
            output.write(f' Like={_quoted(_Like)}')

        _Bus1 = fields.get('Bus1')
        output.write(f' Bus1={_quoted(_Bus1)}')
        _Amps = fields.get('Amps')
        if _Amps is not None:
            output.write(f' Amps={_Amps}')

        _Angle = fields.get('Angle')
        if _Angle is not None:
            output.write(f' Angle={_Angle}')

        _Frequency = fields.get('Frequency')
        if _Frequency is not None:
            output.write(f' Frequency={_Frequency}')

        _Phases = fields.get('Phases')
        if _Phases is not None:
            output.write(f' Phases={_Phases}')

        _ScanType = fields.get('ScanType')
        if _ScanType is not None:
            output.write(f' ScanType={_quoted(_ScanType)}')

        _Sequence = fields.get('Sequence')
        if _Sequence is not None:
            output.write(f' Sequence={_quoted(_Sequence)}')

        _Yearly = fields.get('Yearly')
        if _Yearly is not None:
            output.write(f' Yearly={_quoted(_Yearly)}')

        _Daily = fields.get('Daily')
        if _Daily is not None:
            output.write(f' Daily={_quoted(_Daily)}')

        _Duty = fields.get('Duty')
        if _Duty is not None:
            output.write(f' Duty={_quoted(_Duty)}')

        _Bus2 = fields.get('Bus2')
        if _Bus2 is not None:
            output.write(f' Bus2={_quoted(_Bus2)}')

        _Spectrum = fields.get('Spectrum')
        if _Spectrum is not None:
            output.write(f' Spectrum={_quoted(_Spectrum)}')

        _BaseFreq = fields.get('BaseFreq')
        if _BaseFreq is not None:
            output.write(f' BaseFreq={_BaseFreq}')

        _Enabled = fields.get('Enabled')
        if _Enabled is not None:
            output.write(f' Enabled={_Enabled}')

        output.write('\n')


Isource_ = Isource


class IsourceList(RootModel[List[Isource]]):
    root: List[Isource]





class IsourceContainer(RootModel[Union[IsourceList, JSONFilePath, JSONLinesFilePath]]):
    root: Union[IsourceList, JSONFilePath, JSONLinesFilePath] = Field(..., title="IsourceContainer")


    @model_validator(mode="before")
    def _val_dss_model(self) -> "IsourceContainer":
        try:
            _fields_set = set(self.keys())
        except:
            if isinstance(self, (list, tuple)):
                return self

            raise

        # Validate oneOf (spec. sets) based on the `required` lists
        _required_IsourceList = "root" in _fields_set and isinstance(self.root, IsourceList)
        _required_JSONFilePath = _fields_set.issuperset({'JSONFile'})
        _required_JSONLinesFilePath = _fields_set.issuperset({'JSONLinesFile'})
        num_specs = _required_IsourceList + _required_JSONFilePath + _required_JSONLinesFilePath
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



class VCCS(BaseModel):
    Name: Annotated[str, Field(max_length=255, min_length=1)] = Field(..., title="Name")
    Like: Optional[str] = Field(None, title="Like")
    Bus1: BusConnection = Field(..., title="Bus1")
    Phases: Optional[PositiveInt] = Field(None, title="Phases")
    PRated: Optional[float] = Field(None, title="PRated")
    VRated: Optional[PositiveFloat] = Field(None, title="VRated")
    Ppct: Optional[float] = Field(None, title="Ppct")
    BP1: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="BP1")
    BP2: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="BP2")
    Filter: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="Filter")
    FSample: Optional[PositiveFloat] = Field(None, title="FSample")
    RMSMode: Optional[bool] = Field(None, title="RMSMode")
    IMaxpu: Optional[float] = Field(None, title="IMaxpu")
    VRMSTau: Optional[float] = Field(None, title="VRMSTau")
    IRMSTau: Optional[float] = Field(None, title="IRMSTau")
    Spectrum: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="Spectrum")
    BaseFreq: Optional[PositiveFloat] = Field(None, title="BaseFreq")
    Enabled: Optional[bool] = Field(None, title="Enabled")

    def dumps_dss(self, edit=False) -> str:
        """
        Dump the DSS object dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            self.dump_dss(output, edit)
            return output.getvalue()

    def dump_dss(self, output: TextIO, edit=False):
        """
        Dump the DSS object to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        VCCS.dict_dump_dss(self.model_dump(exclude_unset=True), output, edit=edit)

    @staticmethod
    def dict_dumps_dss(fields: Dict, edit=False) -> str:
        """
        Dump the DSS object from a dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            VCCS.dict_dump_dss(fields, output, edit)
            return output.getvalue()

    @staticmethod
    def dict_dump_dss(fields: Dict, output: TextIO, edit=False):
        """
        Dump the DSS object from a dict to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        if edit:
            output.write(f'''edit VCCS.{fields['Name']}''')
        else:
            output.write(f'''new VCCS.{fields['Name']}''')

        _Like = fields.get('Like')
        if _Like is not None:
            output.write(f' Like={_quoted(_Like)}')

        _Bus1 = fields.get('Bus1')
        output.write(f' Bus1={_quoted(_Bus1)}')
        _Phases = fields.get('Phases')
        if _Phases is not None:
            output.write(f' Phases={_Phases}')

        _PRated = fields.get('PRated')
        if _PRated is not None:
            output.write(f' PRated={_PRated}')

        _VRated = fields.get('VRated')
        if _VRated is not None:
            output.write(f' VRated={_VRated}')

        _Ppct = fields.get('Ppct')
        if _Ppct is not None:
            output.write(f' Ppct={_Ppct}')

        _BP1 = fields.get('BP1')
        if _BP1 is not None:
            output.write(f' BP1={_quoted(_BP1)}')

        _BP2 = fields.get('BP2')
        if _BP2 is not None:
            output.write(f' BP2={_quoted(_BP2)}')

        _Filter = fields.get('Filter')
        if _Filter is not None:
            output.write(f' Filter={_quoted(_Filter)}')

        _FSample = fields.get('FSample')
        if _FSample is not None:
            output.write(f' FSample={_FSample}')

        _RMSMode = fields.get('RMSMode')
        if _RMSMode is not None:
            output.write(f' RMSMode={_RMSMode}')

        _IMaxpu = fields.get('IMaxpu')
        if _IMaxpu is not None:
            output.write(f' IMaxpu={_IMaxpu}')

        _VRMSTau = fields.get('VRMSTau')
        if _VRMSTau is not None:
            output.write(f' VRMSTau={_VRMSTau}')

        _IRMSTau = fields.get('IRMSTau')
        if _IRMSTau is not None:
            output.write(f' IRMSTau={_IRMSTau}')

        _Spectrum = fields.get('Spectrum')
        if _Spectrum is not None:
            output.write(f' Spectrum={_quoted(_Spectrum)}')

        _BaseFreq = fields.get('BaseFreq')
        if _BaseFreq is not None:
            output.write(f' BaseFreq={_BaseFreq}')

        _Enabled = fields.get('Enabled')
        if _Enabled is not None:
            output.write(f' Enabled={_Enabled}')

        output.write('\n')


VCCS_ = VCCS


class VCCSList(RootModel[List[VCCS]]):
    root: List[VCCS]





class VCCSContainer(RootModel[Union[VCCSList, JSONFilePath, JSONLinesFilePath]]):
    root: Union[VCCSList, JSONFilePath, JSONLinesFilePath] = Field(..., title="VCCSContainer")


    @model_validator(mode="before")
    def _val_dss_model(self) -> "VCCSContainer":
        try:
            _fields_set = set(self.keys())
        except:
            if isinstance(self, (list, tuple)):
                return self

            raise

        # Validate oneOf (spec. sets) based on the `required` lists
        _required_VCCSList = "root" in _fields_set and isinstance(self.root, VCCSList)
        _required_JSONFilePath = _fields_set.issuperset({'JSONFile'})
        _required_JSONLinesFilePath = _fields_set.issuperset({'JSONLinesFile'})
        num_specs = _required_VCCSList + _required_JSONFilePath + _required_JSONLinesFilePath
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



class Load_Common(BaseModel):
    Name: Annotated[str, Field(max_length=255, min_length=1)] = Field(..., title="Name")
    Like: Optional[str] = Field(None, title="Like")
    Phases: Optional[PositiveInt] = Field(None, title="Phases")
    Bus1: BusConnection = Field(..., title="Bus1")
    kV: Annotated[float, Field(ge=0)] = Field(..., title="kV")
    Model: Optional[LoadModel] = Field(None, title="Model")
    Yearly: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="Yearly")
    Daily: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="Daily")
    Duty: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="Duty")
    Growth: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="Growth")
    Conn: Optional[Connection] = Field(None, title="Conn")
    RNeut: Optional[float] = Field(None, title="RNeut")
    XNeut: Optional[float] = Field(None, title="XNeut")
    Status: Optional[LoadStatus] = Field(None, title="Status")
    Class: Optional[int] = Field(None, title="Class")
    VMinpu: Optional[float] = Field(None, title="VMinpu")
    VMaxpu: Optional[float] = Field(None, title="VMaxpu")
    VMinNorm: Optional[float] = Field(None, title="VMinNorm")
    VMinEmerg: Optional[float] = Field(None, title="VMinEmerg")
    pctMean: Optional[float] = Field(None, title="%Mean", validation_alias=AliasChoices("pctMean", "%Mean"))
    pctStdDev: Optional[float] = Field(None, title="%StdDev", validation_alias=AliasChoices("pctStdDev", "%StdDev"))
    CVRWatts: Optional[float] = Field(None, title="CVRWatts")
    CVRVars: Optional[float] = Field(None, title="CVRVars")
    CVRCurve: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="CVRCurve")
    NumCust: Optional[int] = Field(None, title="NumCust")
    ZIPV: Optional[List[float]] = Field(None, title="ZIPV")
    pctSeriesRL: Optional[float] = Field(None, title="%SeriesRL", validation_alias=AliasChoices("pctSeriesRL", "%SeriesRL"))
    RelWeight: Optional[float] = Field(None, title="RelWeight")
    VLowpu: Optional[float] = Field(None, title="VLowpu")
    puXHarm: Optional[float] = Field(None, title="puXHarm")
    XRHarm: Optional[float] = Field(None, title="XRHarm")
    Spectrum: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="Spectrum")
    BaseFreq: Optional[PositiveFloat] = Field(None, title="BaseFreq")
    Enabled: Optional[bool] = Field(None, title="Enabled")

class Load_kWPF(Load_Common):
    kW: float = Field(..., title="kW")
    PF: Annotated[float, Field(le=1, ge=-1)] = Field(..., title="PF")

class Load_kWkvar(Load_Common):
    kW: float = Field(..., title="kW")
    kvar: float = Field(..., title="kvar")

class Load_kVAPF(Load_Common):
    PF: Annotated[float, Field(le=1, ge=-1)] = Field(..., title="PF")
    kVA: float = Field(..., title="kVA")

class Load_xfkVAAllocationFactorPF(Load_Common):
    PF: Annotated[float, Field(le=1, ge=-1)] = Field(..., title="PF")
    XfkVA: float = Field(..., title="XfkVA")
    AllocationFactor: Optional[float] = Field(None, title="AllocationFactor")

class Load_kWhkWhDaysCFactorPF(Load_Common):
    PF: Annotated[float, Field(le=1, ge=-1)] = Field(..., title="PF")
    kWh: float = Field(..., title="kWh")
    kWhDays: Optional[float] = Field(None, title="kWhDays")
    CFactor: Optional[float] = Field(None, title="CFactor")


class Load(RootModel[Union[Load_kWPF, Load_kWkvar, Load_kVAPF, Load_xfkVAAllocationFactorPF, Load_kWhkWhDaysCFactorPF]]):
    root: Union[Load_kWPF, Load_kWkvar, Load_kVAPF, Load_xfkVAAllocationFactorPF, Load_kWhkWhDaysCFactorPF] = Field(..., title="Load")

    def dumps_dss(self, edit=False) -> str:
        """
        Dump the DSS object dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            self.dump_dss(output, edit)
            return output.getvalue()

    def dump_dss(self, output: TextIO, edit=False):
        """
        Dump the DSS object to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        Load.dict_dump_dss(self.model_dump(exclude_unset=True), output, edit=edit)

    @staticmethod
    def dict_dumps_dss(fields: Dict, edit=False) -> str:
        """
        Dump the DSS object from a dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            Load.dict_dump_dss(fields, output, edit)
            return output.getvalue()

    @staticmethod
    def dict_dump_dss(fields: Dict, output: TextIO, edit=False):
        """
        Dump the DSS object from a dict to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        if edit:
            output.write(f'''edit Load.{fields['Name']}''')
        else:
            output.write(f'''new Load.{fields['Name']}''')

        _Like = fields.get('Like')
        if _Like is not None:
            output.write(f' Like={_quoted(_Like)}')

        _Phases = fields.get('Phases')
        if _Phases is not None:
            output.write(f' Phases={_Phases}')

        _Bus1 = fields.get('Bus1')
        output.write(f' Bus1={_quoted(_Bus1)}')
        _kV = fields.get('kV')
        output.write(f' kV={_kV}')
        _kW = fields.get('kW')
        if _kW is not None:
            output.write(f' kW={_kW}')

        _PF = fields.get('PF')
        if _PF is not None:
            output.write(f' PF={_PF}')

        _Model = fields.get('Model')
        if _Model is not None:
            output.write(f' Model={_Model}')

        _Yearly = fields.get('Yearly')
        if _Yearly is not None:
            output.write(f' Yearly={_quoted(_Yearly)}')

        _Daily = fields.get('Daily')
        if _Daily is not None:
            output.write(f' Daily={_quoted(_Daily)}')

        _Duty = fields.get('Duty')
        if _Duty is not None:
            output.write(f' Duty={_quoted(_Duty)}')

        _Growth = fields.get('Growth')
        if _Growth is not None:
            output.write(f' Growth={_quoted(_Growth)}')

        _Conn = fields.get('Conn')
        if _Conn is not None:
            output.write(f' Conn={_quoted(_Conn)}')

        _kvar = fields.get('kvar')
        if _kvar is not None:
            output.write(f' kvar={_kvar}')

        _RNeut = fields.get('RNeut')
        if _RNeut is not None:
            output.write(f' RNeut={_RNeut}')

        _XNeut = fields.get('XNeut')
        if _XNeut is not None:
            output.write(f' XNeut={_XNeut}')

        _Status = fields.get('Status')
        if _Status is not None:
            output.write(f' Status={_quoted(_Status)}')

        _Class = fields.get('Class')
        if _Class is not None:
            output.write(f' Class={_Class}')

        _VMinpu = fields.get('VMinpu')
        if _VMinpu is not None:
            output.write(f' VMinpu={_VMinpu}')

        _VMaxpu = fields.get('VMaxpu')
        if _VMaxpu is not None:
            output.write(f' VMaxpu={_VMaxpu}')

        _VMinNorm = fields.get('VMinNorm')
        if _VMinNorm is not None:
            output.write(f' VMinNorm={_VMinNorm}')

        _VMinEmerg = fields.get('VMinEmerg')
        if _VMinEmerg is not None:
            output.write(f' VMinEmerg={_VMinEmerg}')

        _XfkVA = fields.get('XfkVA')
        if _XfkVA is not None:
            output.write(f' XfkVA={_XfkVA}')

        _AllocationFactor = fields.get('AllocationFactor')
        if _AllocationFactor is not None:
            output.write(f' AllocationFactor={_AllocationFactor}')

        _kVA = fields.get('kVA')
        if _kVA is not None:
            output.write(f' kVA={_kVA}')

        _pctMean = fields.get('pctMean')
        if _pctMean is not None:
            output.write(f' %Mean={_pctMean}')

        _pctStdDev = fields.get('pctStdDev')
        if _pctStdDev is not None:
            output.write(f' %StdDev={_pctStdDev}')

        _CVRWatts = fields.get('CVRWatts')
        if _CVRWatts is not None:
            output.write(f' CVRWatts={_CVRWatts}')

        _CVRVars = fields.get('CVRVars')
        if _CVRVars is not None:
            output.write(f' CVRVars={_CVRVars}')

        _kWh = fields.get('kWh')
        if _kWh is not None:
            output.write(f' kWh={_kWh}')

        _kWhDays = fields.get('kWhDays')
        if _kWhDays is not None:
            output.write(f' kWhDays={_kWhDays}')

        _CFactor = fields.get('CFactor')
        if _CFactor is not None:
            output.write(f' CFactor={_CFactor}')

        _CVRCurve = fields.get('CVRCurve')
        if _CVRCurve is not None:
            output.write(f' CVRCurve={_quoted(_CVRCurve)}')

        _NumCust = fields.get('NumCust')
        if _NumCust is not None:
            output.write(f' NumCust={_NumCust}')

        _ZIPV = fields.get('ZIPV')
        if _ZIPV is not None:
            output.write(f' ZIPV={_as_list(_ZIPV)}')

        _pctSeriesRL = fields.get('pctSeriesRL')
        if _pctSeriesRL is not None:
            output.write(f' %SeriesRL={_pctSeriesRL}')

        _RelWeight = fields.get('RelWeight')
        if _RelWeight is not None:
            output.write(f' RelWeight={_RelWeight}')

        _VLowpu = fields.get('VLowpu')
        if _VLowpu is not None:
            output.write(f' VLowpu={_VLowpu}')

        _puXHarm = fields.get('puXHarm')
        if _puXHarm is not None:
            output.write(f' puXHarm={_puXHarm}')

        _XRHarm = fields.get('XRHarm')
        if _XRHarm is not None:
            output.write(f' XRHarm={_XRHarm}')

        _Spectrum = fields.get('Spectrum')
        if _Spectrum is not None:
            output.write(f' Spectrum={_quoted(_Spectrum)}')

        _BaseFreq = fields.get('BaseFreq')
        if _BaseFreq is not None:
            output.write(f' BaseFreq={_BaseFreq}')

        _Enabled = fields.get('Enabled')
        if _Enabled is not None:
            output.write(f' Enabled={_Enabled}')

        output.write('\n')

    @model_validator(mode="before")
    def _val_dss_model(self) -> "Load":
        _fields_set = set(self.keys())
        # Validate oneOf (spec. sets) based on the `required` lists
        _required_kWPF = _fields_set.issuperset({'kW', 'PF'})
        _required_kWkvar = _fields_set.issuperset({'kvar', 'kW'})
        _required_kVAPF = _fields_set.issuperset({'kVA', 'PF'})
        _required_xfkVAAllocationFactorPF = _fields_set.issuperset({'PF', 'XfkVA'})
        _required_kWhkWhDaysCFactorPF = _fields_set.issuperset({'kWh', 'PF'})
        num_specs = _required_kWPF + _required_kWkvar + _required_kVAPF + _required_xfkVAAllocationFactorPF + _required_kWhkWhDaysCFactorPF
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



Load_ = Load


class LoadList(RootModel[List[Load]]):
    root: List[Load]





class LoadContainer(RootModel[Union[LoadList, JSONFilePath, JSONLinesFilePath]]):
    root: Union[LoadList, JSONFilePath, JSONLinesFilePath] = Field(..., title="LoadContainer")


    @model_validator(mode="before")
    def _val_dss_model(self) -> "LoadContainer":
        try:
            _fields_set = set(self.keys())
        except:
            if isinstance(self, (list, tuple)):
                return self

            raise

        # Validate oneOf (spec. sets) based on the `required` lists
        _required_LoadList = "root" in _fields_set and isinstance(self.root, LoadList)
        _required_JSONFilePath = _fields_set.issuperset({'JSONFile'})
        _required_JSONLinesFilePath = _fields_set.issuperset({'JSONLinesFile'})
        num_specs = _required_LoadList + _required_JSONFilePath + _required_JSONLinesFilePath
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



class Transformer_Common(BaseModel):
    Name: Annotated[str, Field(max_length=255, min_length=1)] = Field(..., title="Name")
    Like: Optional[str] = Field(None, title="Like")
    Phases: Optional[PositiveInt] = Field(None, title="Phases")
    Bus: List[BusConnection] = Field(..., title="Bus")
    Conn: Optional[List[Connection]] = Field(None, title="Conn")
    kVA: Optional[List[float]] = Field(None, title="kVA")
    Tap: Optional[List[float]] = Field(None, title="Tap")
    pctR: Optional[List[float]] = Field(None, title="%R", validation_alias=AliasChoices("pctR", "%R"))
    RNeut: Optional[List[float]] = Field(None, title="RNeut")
    XNeut: Optional[List[float]] = Field(None, title="XNeut")
    Thermal: Optional[float] = Field(None, title="Thermal")
    n: Optional[float] = Field(None, title="n")
    m: Optional[float] = Field(None, title="m")
    FLRise: Optional[float] = Field(None, title="FLRise")
    HSRise: Optional[float] = Field(None, title="HSRise")
    pctLoadLoss: Optional[float] = Field(None, title="%LoadLoss", validation_alias=AliasChoices("pctLoadLoss", "%LoadLoss"))
    pctNoLoadLoss: Optional[float] = Field(None, title="%NoLoadLoss", validation_alias=AliasChoices("pctNoLoadLoss", "%NoLoadLoss"))
    NormHkVA: Optional[float] = Field(None, title="NormHkVA")
    EmergHkVA: Optional[float] = Field(None, title="EmergHkVA")
    Sub: Optional[bool] = Field(None, title="Sub")
    MaxTap: Optional[List[float]] = Field(None, title="MaxTap")
    MinTap: Optional[List[float]] = Field(None, title="MinTap")
    NumTaps: Optional[List[int]] = Field(None, title="NumTaps")
    SubName: Optional[str] = Field(None, title="SubName")
    pctIMag: Optional[float] = Field(None, title="%IMag", validation_alias=AliasChoices("pctIMag", "%IMag"))
    ppm_Antifloat: Optional[float] = Field(None, title="ppm_Antifloat")
    Bank: Optional[str] = Field(None, title="Bank")
    XRConst: Optional[bool] = Field(None, title="XRConst")
    LeadLag: Optional[PhaseSequence] = Field(None, title="LeadLag")
    Core: Optional[CoreType] = Field(None, title="Core")
    RDCOhms: Optional[List[float]] = Field(None, title="RDCOhms")
    Ratings: Optional[ArrayOrFilePath] = Field(None, title="Ratings")
    FaultRate: Optional[float] = Field(None, title="FaultRate")
    pctPerm: Optional[float] = Field(None, title="pctPerm")
    Repair: Optional[float] = Field(None, title="Repair")
    BaseFreq: Optional[PositiveFloat] = Field(None, title="BaseFreq")
    Enabled: Optional[bool] = Field(None, title="Enabled")

class Transformer_XfmrCode(Transformer_Common):
    XfmrCode: Annotated[str, Field(max_length=255, min_length=1)] = Field(..., title="XfmrCode")

class Transformer_X12X13X23kV(Transformer_Common):
    kV: List[float] = Field(..., title="kV")
    X12: PositiveFloat = Field(..., title="X12")
    X13: Optional[PositiveFloat] = Field(None, title="X13")
    X23: Optional[PositiveFloat] = Field(None, title="X23")

class Transformer_XscArraykV(Transformer_Common):
    kV: List[float] = Field(..., title="kV")
    XSCArray: List[float] = Field(..., title="XSCArray")


class Transformer(RootModel[Union[Transformer_XfmrCode, Transformer_X12X13X23kV, Transformer_XscArraykV]]):
    root: Union[Transformer_XfmrCode, Transformer_X12X13X23kV, Transformer_XscArraykV] = Field(..., title="Transformer")

    def dumps_dss(self, edit=False) -> str:
        """
        Dump the DSS object dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            self.dump_dss(output, edit)
            return output.getvalue()

    def dump_dss(self, output: TextIO, edit=False):
        """
        Dump the DSS object to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        Transformer.dict_dump_dss(self.model_dump(exclude_unset=True), output, edit=edit)

    @staticmethod
    def dict_dumps_dss(fields: Dict, edit=False) -> str:
        """
        Dump the DSS object from a dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            Transformer.dict_dump_dss(fields, output, edit)
            return output.getvalue()

    @staticmethod
    def dict_dump_dss(fields: Dict, output: TextIO, edit=False):
        """
        Dump the DSS object from a dict to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        _length_Windings = None

        if edit:
            output.write(f'''edit Transformer.{fields['Name']}''')
        else:
            output.write(f'''new Transformer.{fields['Name']}''')

        _Like = fields.get('Like')
        if _Like is not None:
            output.write(f' Like={_quoted(_Like)}')

        _Phases = fields.get('Phases')
        if _Phases is not None:
            output.write(f' Phases={_Phases}')

        _Bus = fields.get('Bus')
        _length_Windings = len(_Bus)
        output.write(f' Windings={_length_Windings}')
        output.write(f' Buses=({_quoted_list(_Bus)})')
        _Conn = fields.get('Conn')
        if _Conn is not None:
            if _length_Windings is None:
                _length_Windings = len(_Conn)
                output.write(f' Windings={_length_Windings}')
            elif len(_Conn) != _length_Windings:
                raise ValueError(f'Array length ({len(_Conn)}) for "Conn" does not match expected length ({_length_Windings})')

            output.write(f' Conns=({_quoted_list(_Conn)})')

        _kV = fields.get('kV')
        if _kV is not None:
            if _length_Windings is None:
                _length_Windings = len(_kV)
                output.write(f' Windings={_length_Windings}')
            elif len(_kV) != _length_Windings:
                raise ValueError(f'Array length ({len(_kV)}) for "kV" does not match expected length ({_length_Windings})')

            output.write(f' kVs={_as_list(_kV)}')

        _kVA = fields.get('kVA')
        if _kVA is not None:
            if _length_Windings is None:
                _length_Windings = len(_kVA)
                output.write(f' Windings={_length_Windings}')
            elif len(_kVA) != _length_Windings:
                raise ValueError(f'Array length ({len(_kVA)}) for "kVA" does not match expected length ({_length_Windings})')

            output.write(f' kVAs={_as_list(_kVA)}')

        _Tap = fields.get('Tap')
        if _Tap is not None:
            if _length_Windings is None:
                _length_Windings = len(_Tap)
                output.write(f' Windings={_length_Windings}')
            elif len(_Tap) != _length_Windings:
                raise ValueError(f'Array length ({len(_Tap)}) for "Tap" does not match expected length ({_length_Windings})')

            output.write(f' Taps={_as_list(_Tap)}')

        _pctR = fields.get('pctR')
        if _pctR is not None:
            if _length_Windings is None:
                _length_Windings = len(_pctR)
                output.write(f' Windings={_length_Windings}')
            elif len(_pctR) != _length_Windings:
                raise ValueError(f'Array length ({len(_pctR)}) for "pctR" does not match expected length ({_length_Windings})')

            output.write(f' %Rs={_as_list(_pctR)}')

        _XSCArray = fields.get('XSCArray')
        if _XSCArray is not None:
            output.write(f' XSCArray={_as_list(_XSCArray)}')

        _Thermal = fields.get('Thermal')
        if _Thermal is not None:
            output.write(f' Thermal={_Thermal}')

        _n = fields.get('n')
        if _n is not None:
            output.write(f' n={_n}')

        _m = fields.get('m')
        if _m is not None:
            output.write(f' m={_m}')

        _FLRise = fields.get('FLRise')
        if _FLRise is not None:
            output.write(f' FLRise={_FLRise}')

        _HSRise = fields.get('HSRise')
        if _HSRise is not None:
            output.write(f' HSRise={_HSRise}')

        _pctLoadLoss = fields.get('pctLoadLoss')
        if _pctLoadLoss is not None:
            output.write(f' %LoadLoss={_pctLoadLoss}')

        _pctNoLoadLoss = fields.get('pctNoLoadLoss')
        if _pctNoLoadLoss is not None:
            output.write(f' %NoLoadLoss={_pctNoLoadLoss}')

        _NormHkVA = fields.get('NormHkVA')
        if _NormHkVA is not None:
            output.write(f' NormHkVA={_NormHkVA}')

        _EmergHkVA = fields.get('EmergHkVA')
        if _EmergHkVA is not None:
            output.write(f' EmergHkVA={_EmergHkVA}')

        _Sub = fields.get('Sub')
        if _Sub is not None:
            output.write(f' Sub={_Sub}')

        _SubName = fields.get('SubName')
        if _SubName is not None:
            output.write(f' SubName={_quoted(_SubName)}')

        _pctIMag = fields.get('pctIMag')
        if _pctIMag is not None:
            output.write(f' %IMag={_pctIMag}')

        _ppm_Antifloat = fields.get('ppm_Antifloat')
        if _ppm_Antifloat is not None:
            output.write(f' ppm_Antifloat={_ppm_Antifloat}')

        _Bank = fields.get('Bank')
        if _Bank is not None:
            output.write(f' Bank={_quoted(_Bank)}')

        _XfmrCode = fields.get('XfmrCode')
        if _XfmrCode is not None:
            output.write(f' XfmrCode={_quoted(_XfmrCode)}')

        _XRConst = fields.get('XRConst')
        if _XRConst is not None:
            output.write(f' XRConst={_XRConst}')

        _X12 = fields.get('X12')
        if _X12 is not None:
            output.write(f' X12={_X12}')

        _X13 = fields.get('X13')
        if _X13 is not None:
            output.write(f' X13={_X13}')

        _X23 = fields.get('X23')
        if _X23 is not None:
            output.write(f' X23={_X23}')

        _LeadLag = fields.get('LeadLag')
        if _LeadLag is not None:
            output.write(f' LeadLag={_quoted(_LeadLag)}')

        _Core = fields.get('Core')
        if _Core is not None:
            output.write(f' Core={_quoted(_Core)}')

        _Ratings = fields.get('Ratings')
        if _Ratings is not None:
            if isinstance(_Ratings, ARRAY_LIKE):
                output.write(f' Seasons={len(_Ratings)}')
                _Ratings = _as_list(_Ratings)
            else:
                _length_Ratings, _Ratings = _filepath_array(_Ratings)
                output.write(f' Seasons={_length_Ratings}')
            output.write(f' Ratings={_Ratings}')

        _FaultRate = fields.get('FaultRate')
        if _FaultRate is not None:
            output.write(f' FaultRate={_FaultRate}')

        _pctPerm = fields.get('pctPerm')
        if _pctPerm is not None:
            output.write(f' pctPerm={_pctPerm}')

        _Repair = fields.get('Repair')
        if _Repair is not None:
            output.write(f' Repair={_Repair}')

        _BaseFreq = fields.get('BaseFreq')
        if _BaseFreq is not None:
            output.write(f' BaseFreq={_BaseFreq}')

        _Enabled = fields.get('Enabled')
        if _Enabled is not None:
            output.write(f' Enabled={_Enabled}')

        _needs_Wdg = False
        _RNeut = fields.get('RNeut')
        if _RNeut is not None:
            if len(_RNeut) != _length_Windings:
                raise ValueError(f'Array length ({len(_RNeut)}) for "RNeut" does not match expected length ({_length_Windings})')
            _needs_Wdg = True
        _XNeut = fields.get('XNeut')
        if _XNeut is not None:
            if len(_XNeut) != _length_Windings:
                raise ValueError(f'Array length ({len(_XNeut)}) for "XNeut" does not match expected length ({_length_Windings})')
            _needs_Wdg = True
        _MaxTap = fields.get('MaxTap')
        if _MaxTap is not None:
            if len(_MaxTap) != _length_Windings:
                raise ValueError(f'Array length ({len(_MaxTap)}) for "MaxTap" does not match expected length ({_length_Windings})')
            _needs_Wdg = True
        _MinTap = fields.get('MinTap')
        if _MinTap is not None:
            if len(_MinTap) != _length_Windings:
                raise ValueError(f'Array length ({len(_MinTap)}) for "MinTap" does not match expected length ({_length_Windings})')
            _needs_Wdg = True
        _NumTaps = fields.get('NumTaps')
        if _NumTaps is not None:
            if len(_NumTaps) != _length_Windings:
                raise ValueError(f'Array length ({len(_NumTaps)}) for "NumTaps" does not match expected length ({_length_Windings})')
            _needs_Wdg = True
        _RDCOhms = fields.get('RDCOhms')
        if _RDCOhms is not None:
            if len(_RDCOhms) != _length_Windings:
                raise ValueError(f'Array length ({len(_RDCOhms)}) for "RDCOhms" does not match expected length ({_length_Windings})')
            _needs_Wdg = True
        if _length_Windings is not None and _needs_Wdg:
            for _Wdg in range(_length_Windings):
                output.write(f" Wdg={_Wdg + 1}")
                _RNeut = fields.get('RNeut')
                if _RNeut is not None:
                    output.write(f" RNeut={_RNeut[_Wdg]}")
                _XNeut = fields.get('XNeut')
                if _XNeut is not None:
                    output.write(f" XNeut={_XNeut[_Wdg]}")
                _MaxTap = fields.get('MaxTap')
                if _MaxTap is not None:
                    output.write(f" MaxTap={_MaxTap[_Wdg]}")
                _MinTap = fields.get('MinTap')
                if _MinTap is not None:
                    output.write(f" MinTap={_MinTap[_Wdg]}")
                _NumTaps = fields.get('NumTaps')
                if _NumTaps is not None:
                    output.write(f" NumTaps={_NumTaps[_Wdg]}")
                _RDCOhms = fields.get('RDCOhms')
                if _RDCOhms is not None:
                    output.write(f" RDCOhms={_RDCOhms[_Wdg]}")
        output.write('\n')

    @model_validator(mode="before")
    def _val_dss_model(self) -> "Transformer":
        _fields_set = set(self.keys())
        # Validate oneOf (spec. sets) based on the `required` lists
        _required_XfmrCode = _fields_set.issuperset({'XfmrCode'})
        _required_X12X13X23kV = _fields_set.issuperset({'kV', 'X12'})
        _required_XscArraykV = _fields_set.issuperset({'kV', 'XSCArray'})
        num_specs = _required_XfmrCode + _required_X12X13X23kV + _required_XscArraykV
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



Transformer_ = Transformer


class TransformerList(RootModel[List[Transformer]]):
    root: List[Transformer]





class TransformerContainer(RootModel[Union[TransformerList, JSONFilePath, JSONLinesFilePath]]):
    root: Union[TransformerList, JSONFilePath, JSONLinesFilePath] = Field(..., title="TransformerContainer")


    @model_validator(mode="before")
    def _val_dss_model(self) -> "TransformerContainer":
        try:
            _fields_set = set(self.keys())
        except:
            if isinstance(self, (list, tuple)):
                return self

            raise

        # Validate oneOf (spec. sets) based on the `required` lists
        _required_TransformerList = "root" in _fields_set and isinstance(self.root, TransformerList)
        _required_JSONFilePath = _fields_set.issuperset({'JSONFile'})
        _required_JSONLinesFilePath = _fields_set.issuperset({'JSONLinesFile'})
        num_specs = _required_TransformerList + _required_JSONFilePath + _required_JSONLinesFilePath
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



class RegControl(BaseModel):
    Name: Annotated[str, Field(max_length=255, min_length=1)] = Field(..., title="Name")
    Like: Optional[str] = Field(None, title="Like")
    Transformer: Annotated[str, Field(max_length=255, min_length=1)] = Field(..., title="Transformer")
    Winding: Optional[int] = Field(None, title="Winding")
    VReg: Optional[float] = Field(None, title="VReg")
    Band: Optional[float] = Field(None, title="Band")
    PTRatio: Optional[float] = Field(None, title="PTRatio")
    CTPrim: Optional[float] = Field(None, title="CTPrim")
    R: Optional[float] = Field(None, title="R")
    X: Optional[float] = Field(None, title="X")
    Bus: Optional[str] = Field(None, title="Bus")
    Delay: Optional[float] = Field(None, title="Delay")
    Reversible: Optional[bool] = Field(None, title="Reversible")
    RevVReg: Optional[float] = Field(None, title="RevVReg")
    RevBand: Optional[float] = Field(None, title="RevBand")
    RevR: Optional[float] = Field(None, title="RevR")
    RevX: Optional[float] = Field(None, title="RevX")
    TapDelay: Optional[float] = Field(None, title="TapDelay")
    DebugTrace: Optional[bool] = Field(None, title="DebugTrace")
    MaxTapChange: Optional[int] = Field(None, title="MaxTapChange")
    InverseTime: Optional[bool] = Field(None, title="InverseTime")
    TapWinding: Optional[int] = Field(None, title="TapWinding")
    VLimit: Optional[float] = Field(None, title="VLimit")
    PTPhase: Optional[RegControlPhaseSelection] = Field(None, title="PTPhase")
    RevThreshold: Optional[float] = Field(None, title="RevThreshold")
    RevDelay: Optional[float] = Field(None, title="RevDelay")
    RevNeutral: Optional[bool] = Field(None, title="RevNeutral")
    EventLog: Optional[bool] = Field(None, title="EventLog")
    RemotePTRatio: Optional[float] = Field(None, title="RemotePTRatio")
    TapNum: Optional[int] = Field(None, title="TapNum")
    LDC_Z: Optional[float] = Field(None, title="LDC_Z")
    Rev_Z: Optional[float] = Field(None, title="Rev_Z")
    Cogen: Optional[bool] = Field(None, title="Cogen")
    BaseFreq: Optional[PositiveFloat] = Field(None, title="BaseFreq")
    Enabled: Optional[bool] = Field(None, title="Enabled")
    Reset: Optional[bool] = Field(None, title="Reset")

    def dumps_dss(self, edit=False) -> str:
        """
        Dump the DSS object dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            self.dump_dss(output, edit)
            return output.getvalue()

    def dump_dss(self, output: TextIO, edit=False):
        """
        Dump the DSS object to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        RegControl.dict_dump_dss(self.model_dump(exclude_unset=True), output, edit=edit)

    @staticmethod
    def dict_dumps_dss(fields: Dict, edit=False) -> str:
        """
        Dump the DSS object from a dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            RegControl.dict_dump_dss(fields, output, edit)
            return output.getvalue()

    @staticmethod
    def dict_dump_dss(fields: Dict, output: TextIO, edit=False):
        """
        Dump the DSS object from a dict to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        if edit:
            output.write(f'''edit RegControl.{fields['Name']}''')
        else:
            output.write(f'''new RegControl.{fields['Name']}''')

        _Like = fields.get('Like')
        if _Like is not None:
            output.write(f' Like={_quoted(_Like)}')

        _Transformer = fields.get('Transformer')
        output.write(f' Transformer={_quoted(_Transformer)}')
        _Winding = fields.get('Winding')
        if _Winding is not None:
            output.write(f' Winding={_Winding}')

        _VReg = fields.get('VReg')
        if _VReg is not None:
            output.write(f' VReg={_VReg}')

        _Band = fields.get('Band')
        if _Band is not None:
            output.write(f' Band={_Band}')

        _PTRatio = fields.get('PTRatio')
        if _PTRatio is not None:
            output.write(f' PTRatio={_PTRatio}')

        _CTPrim = fields.get('CTPrim')
        if _CTPrim is not None:
            output.write(f' CTPrim={_CTPrim}')

        _R = fields.get('R')
        if _R is not None:
            output.write(f' R={_R}')

        _X = fields.get('X')
        if _X is not None:
            output.write(f' X={_X}')

        _Bus = fields.get('Bus')
        if _Bus is not None:
            output.write(f' Bus={_quoted(_Bus)}')

        _Delay = fields.get('Delay')
        if _Delay is not None:
            output.write(f' Delay={_Delay}')

        _Reversible = fields.get('Reversible')
        if _Reversible is not None:
            output.write(f' Reversible={_Reversible}')

        _RevVReg = fields.get('RevVReg')
        if _RevVReg is not None:
            output.write(f' RevVReg={_RevVReg}')

        _RevBand = fields.get('RevBand')
        if _RevBand is not None:
            output.write(f' RevBand={_RevBand}')

        _RevR = fields.get('RevR')
        if _RevR is not None:
            output.write(f' RevR={_RevR}')

        _RevX = fields.get('RevX')
        if _RevX is not None:
            output.write(f' RevX={_RevX}')

        _TapDelay = fields.get('TapDelay')
        if _TapDelay is not None:
            output.write(f' TapDelay={_TapDelay}')

        _DebugTrace = fields.get('DebugTrace')
        if _DebugTrace is not None:
            output.write(f' DebugTrace={_DebugTrace}')

        _MaxTapChange = fields.get('MaxTapChange')
        if _MaxTapChange is not None:
            output.write(f' MaxTapChange={_MaxTapChange}')

        _InverseTime = fields.get('InverseTime')
        if _InverseTime is not None:
            output.write(f' InverseTime={_InverseTime}')

        _TapWinding = fields.get('TapWinding')
        if _TapWinding is not None:
            output.write(f' TapWinding={_TapWinding}')

        _VLimit = fields.get('VLimit')
        if _VLimit is not None:
            output.write(f' VLimit={_VLimit}')

        _PTPhase = fields.get('PTPhase')
        if _PTPhase is not None:
            output.write(f' PTPhase={_PTPhase}')

        _RevThreshold = fields.get('RevThreshold')
        if _RevThreshold is not None:
            output.write(f' RevThreshold={_RevThreshold}')

        _RevDelay = fields.get('RevDelay')
        if _RevDelay is not None:
            output.write(f' RevDelay={_RevDelay}')

        _RevNeutral = fields.get('RevNeutral')
        if _RevNeutral is not None:
            output.write(f' RevNeutral={_RevNeutral}')

        _EventLog = fields.get('EventLog')
        if _EventLog is not None:
            output.write(f' EventLog={_EventLog}')

        _RemotePTRatio = fields.get('RemotePTRatio')
        if _RemotePTRatio is not None:
            output.write(f' RemotePTRatio={_RemotePTRatio}')

        _TapNum = fields.get('TapNum')
        if _TapNum is not None:
            output.write(f' TapNum={_TapNum}')

        _LDC_Z = fields.get('LDC_Z')
        if _LDC_Z is not None:
            output.write(f' LDC_Z={_LDC_Z}')

        _Rev_Z = fields.get('Rev_Z')
        if _Rev_Z is not None:
            output.write(f' Rev_Z={_Rev_Z}')

        _Cogen = fields.get('Cogen')
        if _Cogen is not None:
            output.write(f' Cogen={_Cogen}')

        _BaseFreq = fields.get('BaseFreq')
        if _BaseFreq is not None:
            output.write(f' BaseFreq={_BaseFreq}')

        _Enabled = fields.get('Enabled')
        if _Enabled is not None:
            output.write(f' Enabled={_Enabled}')

        _Reset = fields.get('Reset')
        if _Reset is not None:
            output.write(f' Reset={_Reset}')

        output.write('\n')


RegControl_ = RegControl


class RegControlList(RootModel[List[RegControl]]):
    root: List[RegControl]





class RegControlContainer(RootModel[Union[RegControlList, JSONFilePath, JSONLinesFilePath]]):
    root: Union[RegControlList, JSONFilePath, JSONLinesFilePath] = Field(..., title="RegControlContainer")


    @model_validator(mode="before")
    def _val_dss_model(self) -> "RegControlContainer":
        try:
            _fields_set = set(self.keys())
        except:
            if isinstance(self, (list, tuple)):
                return self

            raise

        # Validate oneOf (spec. sets) based on the `required` lists
        _required_RegControlList = "root" in _fields_set and isinstance(self.root, RegControlList)
        _required_JSONFilePath = _fields_set.issuperset({'JSONFile'})
        _required_JSONLinesFilePath = _fields_set.issuperset({'JSONLinesFile'})
        num_specs = _required_RegControlList + _required_JSONFilePath + _required_JSONLinesFilePath
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



class Capacitor_Common(BaseModel):
    Name: Annotated[str, Field(max_length=255, min_length=1)] = Field(..., title="Name")
    Like: Optional[str] = Field(None, title="Like")
    Bus1: BusConnection = Field(..., title="Bus1")
    Bus2: Optional[BusConnection] = Field(None, title="Bus2")
    Phases: Optional[PositiveInt] = Field(None, title="Phases")
    Conn: Optional[Connection] = Field(None, title="Conn")
    R: Optional[ArrayOrFilePath] = Field(None, title="R")
    XL: Optional[ArrayOrFilePath] = Field(None, title="XL")
    Harm: Optional[ArrayOrFilePath] = Field(None, title="Harm")
    States: Optional[List[int]] = Field(None, title="States")
    NormAmps: Optional[float] = Field(None, title="NormAmps")
    EmergAmps: Optional[float] = Field(None, title="EmergAmps")
    FaultRate: Optional[float] = Field(None, title="FaultRate")
    pctPerm: Optional[float] = Field(None, title="pctPerm")
    Repair: Optional[float] = Field(None, title="Repair")
    BaseFreq: Optional[PositiveFloat] = Field(None, title="BaseFreq")
    Enabled: Optional[bool] = Field(None, title="Enabled")

class Capacitor_kvarkV(Capacitor_Common):
    kvar: ArrayOrFilePath = Field(..., title="kvar")
    kV: Annotated[float, Field(ge=0)] = Field(..., title="kV")

class Capacitor_cmatrix(Capacitor_Common):
    CMatrix: SymmetricMatrix = Field(..., title="CMatrix")

    @field_validator('CMatrix')
    @classmethod
    def _CMatrix_check_symm_matrix(cls, value: Optional[SymmetricMatrix], info: ValidationInfo) -> Optional[SymmetricMatrix]:
        return _check_symmetric_matrix(value, info.data.get('Phases'))

class Capacitor_cufkV(Capacitor_Common):
    kV: Annotated[float, Field(ge=0)] = Field(..., title="kV")
    Cuf: ArrayOrFilePath = Field(..., title="Cuf")


class Capacitor(RootModel[Union[Capacitor_kvarkV, Capacitor_cmatrix, Capacitor_cufkV]]):
    root: Union[Capacitor_kvarkV, Capacitor_cmatrix, Capacitor_cufkV] = Field(..., title="Capacitor")

    def dumps_dss(self, edit=False) -> str:
        """
        Dump the DSS object dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            self.dump_dss(output, edit)
            return output.getvalue()

    def dump_dss(self, output: TextIO, edit=False):
        """
        Dump the DSS object to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        Capacitor.dict_dump_dss(self.model_dump(exclude_unset=True), output, edit=edit)

    @staticmethod
    def dict_dumps_dss(fields: Dict, edit=False) -> str:
        """
        Dump the DSS object from a dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            Capacitor.dict_dump_dss(fields, output, edit)
            return output.getvalue()

    @staticmethod
    def dict_dump_dss(fields: Dict, output: TextIO, edit=False):
        """
        Dump the DSS object from a dict to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        _length_NumSteps = None

        if edit:
            output.write(f'''edit Capacitor.{fields['Name']}''')
        else:
            output.write(f'''new Capacitor.{fields['Name']}''')

        _Like = fields.get('Like')
        if _Like is not None:
            output.write(f' Like={_quoted(_Like)}')

        _Bus1 = fields.get('Bus1')
        output.write(f' Bus1={_quoted(_Bus1)}')
        _Bus2 = fields.get('Bus2')
        if _Bus2 is not None:
            output.write(f' Bus2={_quoted(_Bus2)}')

        _Phases = fields.get('Phases')
        if _Phases is not None:
            output.write(f' Phases={_Phases}')

        _kvar = fields.get('kvar')
        if _kvar is not None:
            if isinstance(_kvar, ARRAY_LIKE):
                _length_NumSteps = len(_kvar)
                output.write(f' NumSteps={_length_NumSteps}')
                _kvar = _as_list(_kvar)
            else:
                _length_kvar, _kvar = _filepath_array(_kvar)
                if _length_NumSteps is None:
                    _length_NumSteps = _length_kvar
                    output.write(f' NumSteps={_length_NumSteps}')
                elif _length_NumSteps != _length_kvar:
                    raise ValueError(f'Array length ({_length_kvar}) for "kvar" (from file) does not match expected length ({_length_NumSteps})')

            output.write(f' kvar={_kvar}')

        _kV = fields.get('kV')
        if _kV is not None:
            output.write(f' kV={_kV}')

        _Conn = fields.get('Conn')
        if _Conn is not None:
            output.write(f' Conn={_quoted(_Conn)}')

        _CMatrix = fields.get('CMatrix')
        if _CMatrix is not None:
            output.write(_dump_symmetric_matrix("CMatrix", _CMatrix))

        _Cuf = fields.get('Cuf')
        if _Cuf is not None:
            if isinstance(_Cuf, ARRAY_LIKE):
                if _length_NumSteps is None:
                    _length_NumSteps = len(_Cuf)
                    output.write(f' NumSteps={_length_NumSteps}')
                elif len(_Cuf) != _length_NumSteps:
                    raise ValueError(f'Array length ({len(_Cuf)}) for "Cuf" does not match expected length ({_length_NumSteps})')

                _Cuf = _as_list(_Cuf)
            else:
                _length_Cuf, _Cuf = _filepath_array(_Cuf)
                if _length_NumSteps is None:
                    _length_NumSteps = _length_Cuf
                    output.write(f' NumSteps={_length_NumSteps}')
                elif _length_NumSteps != _length_Cuf:
                    raise ValueError(f'Array length ({_length_Cuf}) for "Cuf" (from file) does not match expected length ({_length_NumSteps})')

            output.write(f' Cuf={_Cuf}')

        _R = fields.get('R')
        if _R is not None:
            if isinstance(_R, ARRAY_LIKE):
                if _length_NumSteps is None:
                    _length_NumSteps = len(_R)
                    output.write(f' NumSteps={_length_NumSteps}')
                elif len(_R) != _length_NumSteps:
                    raise ValueError(f'Array length ({len(_R)}) for "R" does not match expected length ({_length_NumSteps})')

                _R = _as_list(_R)
            else:
                _length_R, _R = _filepath_array(_R)
                if _length_NumSteps is None:
                    _length_NumSteps = _length_R
                    output.write(f' NumSteps={_length_NumSteps}')
                elif _length_NumSteps != _length_R:
                    raise ValueError(f'Array length ({_length_R}) for "R" (from file) does not match expected length ({_length_NumSteps})')

            output.write(f' R={_R}')

        _XL = fields.get('XL')
        if _XL is not None:
            if isinstance(_XL, ARRAY_LIKE):
                if _length_NumSteps is None:
                    _length_NumSteps = len(_XL)
                    output.write(f' NumSteps={_length_NumSteps}')
                elif len(_XL) != _length_NumSteps:
                    raise ValueError(f'Array length ({len(_XL)}) for "XL" does not match expected length ({_length_NumSteps})')

                _XL = _as_list(_XL)
            else:
                _length_XL, _XL = _filepath_array(_XL)
                if _length_NumSteps is None:
                    _length_NumSteps = _length_XL
                    output.write(f' NumSteps={_length_NumSteps}')
                elif _length_NumSteps != _length_XL:
                    raise ValueError(f'Array length ({_length_XL}) for "XL" (from file) does not match expected length ({_length_NumSteps})')

            output.write(f' XL={_XL}')

        _Harm = fields.get('Harm')
        if _Harm is not None:
            if isinstance(_Harm, ARRAY_LIKE):
                if _length_NumSteps is None:
                    _length_NumSteps = len(_Harm)
                    output.write(f' NumSteps={_length_NumSteps}')
                elif len(_Harm) != _length_NumSteps:
                    raise ValueError(f'Array length ({len(_Harm)}) for "Harm" does not match expected length ({_length_NumSteps})')

                _Harm = _as_list(_Harm)
            else:
                _length_Harm, _Harm = _filepath_array(_Harm)
                if _length_NumSteps is None:
                    _length_NumSteps = _length_Harm
                    output.write(f' NumSteps={_length_NumSteps}')
                elif _length_NumSteps != _length_Harm:
                    raise ValueError(f'Array length ({_length_Harm}) for "Harm" (from file) does not match expected length ({_length_NumSteps})')

            output.write(f' Harm={_Harm}')

        _States = fields.get('States')
        if _States is not None:
            if _length_NumSteps is None:
                _length_NumSteps = len(_States)
                output.write(f' NumSteps={_length_NumSteps}')
            elif len(_States) != _length_NumSteps:
                raise ValueError(f'Array length ({len(_States)}) for "States" does not match expected length ({_length_NumSteps})')

            output.write(f' States={_as_list(_States)}')

        _NormAmps = fields.get('NormAmps')
        if _NormAmps is not None:
            output.write(f' NormAmps={_NormAmps}')

        _EmergAmps = fields.get('EmergAmps')
        if _EmergAmps is not None:
            output.write(f' EmergAmps={_EmergAmps}')

        _FaultRate = fields.get('FaultRate')
        if _FaultRate is not None:
            output.write(f' FaultRate={_FaultRate}')

        _pctPerm = fields.get('pctPerm')
        if _pctPerm is not None:
            output.write(f' pctPerm={_pctPerm}')

        _Repair = fields.get('Repair')
        if _Repair is not None:
            output.write(f' Repair={_Repair}')

        _BaseFreq = fields.get('BaseFreq')
        if _BaseFreq is not None:
            output.write(f' BaseFreq={_BaseFreq}')

        _Enabled = fields.get('Enabled')
        if _Enabled is not None:
            output.write(f' Enabled={_Enabled}')

        output.write('\n')

    @model_validator(mode="before")
    def _val_dss_model(self) -> "Capacitor":
        _fields_set = set(self.keys())
        # Validate oneOf (spec. sets) based on the `required` lists
        _required_kvarkV = _fields_set.issuperset({'kV', 'kvar'})
        _required_cmatrix = _fields_set.issuperset({'CMatrix'})
        _required_cufkV = _fields_set.issuperset({'Cuf', 'kV'})
        num_specs = _required_kvarkV + _required_cmatrix + _required_cufkV
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



Capacitor_ = Capacitor


class CapacitorList(RootModel[List[Capacitor]]):
    root: List[Capacitor]





class CapacitorContainer(RootModel[Union[CapacitorList, JSONFilePath, JSONLinesFilePath]]):
    root: Union[CapacitorList, JSONFilePath, JSONLinesFilePath] = Field(..., title="CapacitorContainer")


    @model_validator(mode="before")
    def _val_dss_model(self) -> "CapacitorContainer":
        try:
            _fields_set = set(self.keys())
        except:
            if isinstance(self, (list, tuple)):
                return self

            raise

        # Validate oneOf (spec. sets) based on the `required` lists
        _required_CapacitorList = "root" in _fields_set and isinstance(self.root, CapacitorList)
        _required_JSONFilePath = _fields_set.issuperset({'JSONFile'})
        _required_JSONLinesFilePath = _fields_set.issuperset({'JSONLinesFile'})
        num_specs = _required_CapacitorList + _required_JSONFilePath + _required_JSONLinesFilePath
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



class Reactor_Common(BaseModel):
    Name: Annotated[str, Field(max_length=255, min_length=1)] = Field(..., title="Name")
    Like: Optional[str] = Field(None, title="Like")
    Bus1: BusConnection = Field(..., title="Bus1")
    Bus2: Optional[BusConnection] = Field(None, title="Bus2")
    Phases: Optional[PositiveInt] = Field(None, title="Phases")
    Conn: Optional[Connection] = Field(None, title="Conn")
    Parallel: Optional[bool] = Field(None, title="Parallel")
    Rp: Optional[float] = Field(None, title="Rp")
    NormAmps: Optional[float] = Field(None, title="NormAmps")
    EmergAmps: Optional[float] = Field(None, title="EmergAmps")
    FaultRate: Optional[float] = Field(None, title="FaultRate")
    pctPerm: Optional[float] = Field(None, title="pctPerm")
    Repair: Optional[float] = Field(None, title="Repair")
    BaseFreq: Optional[PositiveFloat] = Field(None, title="BaseFreq")
    Enabled: Optional[bool] = Field(None, title="Enabled")

class Reactor_kVkvarRcurveLcurve(Reactor_Common):
    kvar: float = Field(..., title="kvar")
    kV: Annotated[float, Field(ge=0)] = Field(..., title="kV")
    RCurve: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="RCurve")
    LCurve: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="LCurve")

class Reactor_ZRcurveLcurve(Reactor_Common):
    Z: Complex = Field(..., title="Z")
    RCurve: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="RCurve")
    LCurve: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="LCurve")

class Reactor_RLmHRcurveLcurve(Reactor_Common):
    Z1: Optional[Complex] = Field(None, title="Z1")
    Z2: Optional[Complex] = Field(None, title="Z2")
    Z0: Complex = Field(..., title="Z0")

class Reactor_Z0Z1Z2(Reactor_Common):
    RMatrix: Optional[SymmetricMatrix] = Field(None, title="RMatrix")
    XMatrix: SymmetricMatrix = Field(..., title="XMatrix")

    @field_validator('RMatrix')
    @classmethod
    def _RMatrix_check_symm_matrix(cls, value: Optional[SymmetricMatrix], info: ValidationInfo) -> Optional[SymmetricMatrix]:
        return _check_symmetric_matrix(value, info.data.get('Phases'))

    @field_validator('XMatrix')
    @classmethod
    def _XMatrix_check_symm_matrix(cls, value: Optional[SymmetricMatrix], info: ValidationInfo) -> Optional[SymmetricMatrix]:
        return _check_symmetric_matrix(value, info.data.get('Phases'))


class Reactor(RootModel[Union[Reactor_kVkvarRcurveLcurve, Reactor_ZRcurveLcurve, Reactor_RLmHRcurveLcurve, Reactor_Z0Z1Z2]]):
    root: Union[Reactor_kVkvarRcurveLcurve, Reactor_ZRcurveLcurve, Reactor_RLmHRcurveLcurve, Reactor_Z0Z1Z2] = Field(..., title="Reactor")

    def dumps_dss(self, edit=False) -> str:
        """
        Dump the DSS object dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            self.dump_dss(output, edit)
            return output.getvalue()

    def dump_dss(self, output: TextIO, edit=False):
        """
        Dump the DSS object to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        Reactor.dict_dump_dss(self.model_dump(exclude_unset=True), output, edit=edit)

    @staticmethod
    def dict_dumps_dss(fields: Dict, edit=False) -> str:
        """
        Dump the DSS object from a dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            Reactor.dict_dump_dss(fields, output, edit)
            return output.getvalue()

    @staticmethod
    def dict_dump_dss(fields: Dict, output: TextIO, edit=False):
        """
        Dump the DSS object from a dict to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        if edit:
            output.write(f'''edit Reactor.{fields['Name']}''')
        else:
            output.write(f'''new Reactor.{fields['Name']}''')

        _Like = fields.get('Like')
        if _Like is not None:
            output.write(f' Like={_quoted(_Like)}')

        _Bus1 = fields.get('Bus1')
        output.write(f' Bus1={_quoted(_Bus1)}')
        _Bus2 = fields.get('Bus2')
        if _Bus2 is not None:
            output.write(f' Bus2={_quoted(_Bus2)}')

        _Phases = fields.get('Phases')
        if _Phases is not None:
            output.write(f' Phases={_Phases}')

        _kvar = fields.get('kvar')
        if _kvar is not None:
            output.write(f' kvar={_kvar}')

        _kV = fields.get('kV')
        if _kV is not None:
            output.write(f' kV={_kV}')

        _Conn = fields.get('Conn')
        if _Conn is not None:
            output.write(f' Conn={_quoted(_Conn)}')

        _RMatrix = fields.get('RMatrix')
        if _RMatrix is not None:
            output.write(_dump_symmetric_matrix("RMatrix", _RMatrix))

        _XMatrix = fields.get('XMatrix')
        if _XMatrix is not None:
            output.write(_dump_symmetric_matrix("XMatrix", _XMatrix))

        _Parallel = fields.get('Parallel')
        if _Parallel is not None:
            output.write(f' Parallel={_Parallel}')

        _Rp = fields.get('Rp')
        if _Rp is not None:
            output.write(f' Rp={_Rp}')

        _Z1 = fields.get('Z1')
        if _Z1 is not None:
            output.write(f' Z1={_complex_to_list(_Z1)}')

        _Z2 = fields.get('Z2')
        if _Z2 is not None:
            output.write(f' Z2={_complex_to_list(_Z2)}')

        _Z0 = fields.get('Z0')
        if _Z0 is not None:
            output.write(f' Z0={_complex_to_list(_Z0)}')

        _Z = fields.get('Z')
        if _Z is not None:
            output.write(f' Z={_complex_to_list(_Z)}')

        _RCurve = fields.get('RCurve')
        if _RCurve is not None:
            output.write(f' RCurve={_quoted(_RCurve)}')

        _LCurve = fields.get('LCurve')
        if _LCurve is not None:
            output.write(f' LCurve={_quoted(_LCurve)}')

        _NormAmps = fields.get('NormAmps')
        if _NormAmps is not None:
            output.write(f' NormAmps={_NormAmps}')

        _EmergAmps = fields.get('EmergAmps')
        if _EmergAmps is not None:
            output.write(f' EmergAmps={_EmergAmps}')

        _FaultRate = fields.get('FaultRate')
        if _FaultRate is not None:
            output.write(f' FaultRate={_FaultRate}')

        _pctPerm = fields.get('pctPerm')
        if _pctPerm is not None:
            output.write(f' pctPerm={_pctPerm}')

        _Repair = fields.get('Repair')
        if _Repair is not None:
            output.write(f' Repair={_Repair}')

        _BaseFreq = fields.get('BaseFreq')
        if _BaseFreq is not None:
            output.write(f' BaseFreq={_BaseFreq}')

        _Enabled = fields.get('Enabled')
        if _Enabled is not None:
            output.write(f' Enabled={_Enabled}')

        output.write('\n')

    @model_validator(mode="before")
    def _val_dss_model(self) -> "Reactor":
        _fields_set = set(self.keys())
        # Validate oneOf (spec. sets) based on the `required` lists
        _required_kVkvarRcurveLcurve = _fields_set.issuperset({'kV', 'kvar'})
        _required_ZRcurveLcurve = _fields_set.issuperset({'Z'})
        _required_RLmHRcurveLcurve = _fields_set.issuperset({'Z0'})
        _required_Z0Z1Z2 = _fields_set.issuperset({'XMatrix'})
        num_specs = _required_kVkvarRcurveLcurve + _required_ZRcurveLcurve + _required_RLmHRcurveLcurve + _required_Z0Z1Z2
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



Reactor_ = Reactor


class ReactorList(RootModel[List[Reactor]]):
    root: List[Reactor]





class ReactorContainer(RootModel[Union[ReactorList, JSONFilePath, JSONLinesFilePath]]):
    root: Union[ReactorList, JSONFilePath, JSONLinesFilePath] = Field(..., title="ReactorContainer")


    @model_validator(mode="before")
    def _val_dss_model(self) -> "ReactorContainer":
        try:
            _fields_set = set(self.keys())
        except:
            if isinstance(self, (list, tuple)):
                return self

            raise

        # Validate oneOf (spec. sets) based on the `required` lists
        _required_ReactorList = "root" in _fields_set and isinstance(self.root, ReactorList)
        _required_JSONFilePath = _fields_set.issuperset({'JSONFile'})
        _required_JSONLinesFilePath = _fields_set.issuperset({'JSONLinesFile'})
        num_specs = _required_ReactorList + _required_JSONFilePath + _required_JSONLinesFilePath
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



class CapControl(BaseModel):
    Name: Annotated[str, Field(max_length=255, min_length=1)] = Field(..., title="Name")
    Like: Optional[str] = Field(None, title="Like")
    Element: Optional[str] = Field(None, title="Element")
    Terminal: Optional[int] = Field(None, title="Terminal")
    Capacitor: Annotated[str, Field(max_length=255, min_length=1)] = Field(..., title="Capacitor")
    Type: Optional[CapControlType] = Field(None, title="Type")
    PTRatio: Optional[float] = Field(None, title="PTRatio")
    CTRatio: Optional[float] = Field(None, title="CTRatio")
    OnSetting: Optional[float] = Field(None, title="OnSetting")
    OffSetting: Optional[float] = Field(None, title="OffSetting")
    Delay: Optional[float] = Field(None, title="Delay")
    VoltOverride: Optional[bool] = Field(None, title="VoltOverride")
    VMax: Optional[float] = Field(None, title="VMax")
    VMin: Optional[float] = Field(None, title="VMin")
    DelayOff: Optional[float] = Field(None, title="DelayOff")
    DeadTime: Optional[float] = Field(None, title="DeadTime")
    CTPhase: Optional[MonitoredPhase] = Field(None, title="CTPhase")
    PTPhase: Optional[MonitoredPhase] = Field(None, title="PTPhase")
    VBus: Optional[str] = Field(None, title="VBus")
    EventLog: Optional[bool] = Field(None, title="EventLog")
    UserModel: Optional[FilePath] = Field(None, title="UserModel")
    UserData: Optional[str] = Field(None, title="UserData")
    pctMinkvar: Optional[float] = Field(None, title="pctMinkvar")
    ControlSignal: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="ControlSignal")
    BaseFreq: Optional[PositiveFloat] = Field(None, title="BaseFreq")
    Enabled: Optional[bool] = Field(None, title="Enabled")
    Reset: Optional[bool] = Field(None, title="Reset")

    def dumps_dss(self, edit=False) -> str:
        """
        Dump the DSS object dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            self.dump_dss(output, edit)
            return output.getvalue()

    def dump_dss(self, output: TextIO, edit=False):
        """
        Dump the DSS object to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        CapControl.dict_dump_dss(self.model_dump(exclude_unset=True), output, edit=edit)

    @staticmethod
    def dict_dumps_dss(fields: Dict, edit=False) -> str:
        """
        Dump the DSS object from a dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            CapControl.dict_dump_dss(fields, output, edit)
            return output.getvalue()

    @staticmethod
    def dict_dump_dss(fields: Dict, output: TextIO, edit=False):
        """
        Dump the DSS object from a dict to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        if edit:
            output.write(f'''edit CapControl.{fields['Name']}''')
        else:
            output.write(f'''new CapControl.{fields['Name']}''')

        _Like = fields.get('Like')
        if _Like is not None:
            output.write(f' Like={_quoted(_Like)}')

        _Element = fields.get('Element')
        if _Element is not None:
            output.write(f' Element={_quoted(_Element)}')

        _Terminal = fields.get('Terminal')
        if _Terminal is not None:
            output.write(f' Terminal={_Terminal}')

        _Capacitor = fields.get('Capacitor')
        output.write(f' Capacitor={_quoted(_Capacitor)}')
        _Type = fields.get('Type')
        if _Type is not None:
            output.write(f' Type={_quoted(_Type)}')

        _PTRatio = fields.get('PTRatio')
        if _PTRatio is not None:
            output.write(f' PTRatio={_PTRatio}')

        _CTRatio = fields.get('CTRatio')
        if _CTRatio is not None:
            output.write(f' CTRatio={_CTRatio}')

        _OnSetting = fields.get('OnSetting')
        if _OnSetting is not None:
            output.write(f' OnSetting={_OnSetting}')

        _OffSetting = fields.get('OffSetting')
        if _OffSetting is not None:
            output.write(f' OffSetting={_OffSetting}')

        _Delay = fields.get('Delay')
        if _Delay is not None:
            output.write(f' Delay={_Delay}')

        _VoltOverride = fields.get('VoltOverride')
        if _VoltOverride is not None:
            output.write(f' VoltOverride={_VoltOverride}')

        _VMax = fields.get('VMax')
        if _VMax is not None:
            output.write(f' VMax={_VMax}')

        _VMin = fields.get('VMin')
        if _VMin is not None:
            output.write(f' VMin={_VMin}')

        _DelayOff = fields.get('DelayOff')
        if _DelayOff is not None:
            output.write(f' DelayOff={_DelayOff}')

        _DeadTime = fields.get('DeadTime')
        if _DeadTime is not None:
            output.write(f' DeadTime={_DeadTime}')

        _CTPhase = fields.get('CTPhase')
        if _CTPhase is not None:
            output.write(f' CTPhase={_CTPhase}')

        _PTPhase = fields.get('PTPhase')
        if _PTPhase is not None:
            output.write(f' PTPhase={_PTPhase}')

        _VBus = fields.get('VBus')
        if _VBus is not None:
            output.write(f' VBus={_quoted(_VBus)}')

        _EventLog = fields.get('EventLog')
        if _EventLog is not None:
            output.write(f' EventLog={_EventLog}')

        _UserModel = fields.get('UserModel')
        if _UserModel is not None:
            output.write(f' UserModel={_quoted(_UserModel)}')

        _UserData = fields.get('UserData')
        if _UserData is not None:
            output.write(f' UserData={_quoted(_UserData)}')

        _pctMinkvar = fields.get('pctMinkvar')
        if _pctMinkvar is not None:
            output.write(f' pctMinkvar={_pctMinkvar}')

        _ControlSignal = fields.get('ControlSignal')
        if _ControlSignal is not None:
            output.write(f' ControlSignal={_quoted(_ControlSignal)}')

        _BaseFreq = fields.get('BaseFreq')
        if _BaseFreq is not None:
            output.write(f' BaseFreq={_BaseFreq}')

        _Enabled = fields.get('Enabled')
        if _Enabled is not None:
            output.write(f' Enabled={_Enabled}')

        _Reset = fields.get('Reset')
        if _Reset is not None:
            output.write(f' Reset={_Reset}')

        output.write('\n')


CapControl_ = CapControl


class CapControlList(RootModel[List[CapControl]]):
    root: List[CapControl]





class CapControlContainer(RootModel[Union[CapControlList, JSONFilePath, JSONLinesFilePath]]):
    root: Union[CapControlList, JSONFilePath, JSONLinesFilePath] = Field(..., title="CapControlContainer")


    @model_validator(mode="before")
    def _val_dss_model(self) -> "CapControlContainer":
        try:
            _fields_set = set(self.keys())
        except:
            if isinstance(self, (list, tuple)):
                return self

            raise

        # Validate oneOf (spec. sets) based on the `required` lists
        _required_CapControlList = "root" in _fields_set and isinstance(self.root, CapControlList)
        _required_JSONFilePath = _fields_set.issuperset({'JSONFile'})
        _required_JSONLinesFilePath = _fields_set.issuperset({'JSONLinesFile'})
        num_specs = _required_CapControlList + _required_JSONFilePath + _required_JSONLinesFilePath
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



class Fault_Common(BaseModel):
    Name: Annotated[str, Field(max_length=255, min_length=1)] = Field(..., title="Name")
    Like: Optional[str] = Field(None, title="Like")
    Bus1: BusConnection = Field(..., title="Bus1")
    Bus2: Optional[BusConnection] = Field(None, title="Bus2")
    Phases: Optional[PositiveInt] = Field(None, title="Phases")
    pctStdDev: Optional[float] = Field(None, title="%StdDev", validation_alias=AliasChoices("pctStdDev", "%StdDev"))
    OnTime: Optional[float] = Field(None, title="OnTime")
    Temporary: Optional[bool] = Field(None, title="Temporary")
    MinAmps: Optional[float] = Field(None, title="MinAmps")
    FaultRate: Optional[float] = Field(None, title="FaultRate")
    pctPerm: Optional[float] = Field(None, title="pctPerm")
    Repair: Optional[float] = Field(None, title="Repair")
    BaseFreq: Optional[PositiveFloat] = Field(None, title="BaseFreq")
    Enabled: Optional[bool] = Field(None, title="Enabled")

class Fault_r(Fault_Common):
    R: float = Field(..., title="R")

class Fault_Gmatrix(Fault_Common):
    GMatrix: SymmetricMatrix = Field(..., title="GMatrix")

    @field_validator('GMatrix')
    @classmethod
    def _GMatrix_check_symm_matrix(cls, value: Optional[SymmetricMatrix], info: ValidationInfo) -> Optional[SymmetricMatrix]:
        return _check_symmetric_matrix(value, info.data.get('Phases'))


class Fault(RootModel[Union[Fault_r, Fault_Gmatrix]]):
    root: Union[Fault_r, Fault_Gmatrix] = Field(..., title="Fault")

    def dumps_dss(self, edit=False) -> str:
        """
        Dump the DSS object dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            self.dump_dss(output, edit)
            return output.getvalue()

    def dump_dss(self, output: TextIO, edit=False):
        """
        Dump the DSS object to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        Fault.dict_dump_dss(self.model_dump(exclude_unset=True), output, edit=edit)

    @staticmethod
    def dict_dumps_dss(fields: Dict, edit=False) -> str:
        """
        Dump the DSS object from a dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            Fault.dict_dump_dss(fields, output, edit)
            return output.getvalue()

    @staticmethod
    def dict_dump_dss(fields: Dict, output: TextIO, edit=False):
        """
        Dump the DSS object from a dict to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        if edit:
            output.write(f'''edit Fault.{fields['Name']}''')
        else:
            output.write(f'''new Fault.{fields['Name']}''')

        _Like = fields.get('Like')
        if _Like is not None:
            output.write(f' Like={_quoted(_Like)}')

        _Bus1 = fields.get('Bus1')
        output.write(f' Bus1={_quoted(_Bus1)}')
        _Bus2 = fields.get('Bus2')
        if _Bus2 is not None:
            output.write(f' Bus2={_quoted(_Bus2)}')

        _Phases = fields.get('Phases')
        if _Phases is not None:
            output.write(f' Phases={_Phases}')

        _R = fields.get('R')
        if _R is not None:
            output.write(f' R={_R}')

        _pctStdDev = fields.get('pctStdDev')
        if _pctStdDev is not None:
            output.write(f' %StdDev={_pctStdDev}')

        _GMatrix = fields.get('GMatrix')
        if _GMatrix is not None:
            output.write(_dump_symmetric_matrix("GMatrix", _GMatrix))

        _OnTime = fields.get('OnTime')
        if _OnTime is not None:
            output.write(f' OnTime={_OnTime}')

        _Temporary = fields.get('Temporary')
        if _Temporary is not None:
            output.write(f' Temporary={_Temporary}')

        _MinAmps = fields.get('MinAmps')
        if _MinAmps is not None:
            output.write(f' MinAmps={_MinAmps}')

        _FaultRate = fields.get('FaultRate')
        if _FaultRate is not None:
            output.write(f' FaultRate={_FaultRate}')

        _pctPerm = fields.get('pctPerm')
        if _pctPerm is not None:
            output.write(f' pctPerm={_pctPerm}')

        _Repair = fields.get('Repair')
        if _Repair is not None:
            output.write(f' Repair={_Repair}')

        _BaseFreq = fields.get('BaseFreq')
        if _BaseFreq is not None:
            output.write(f' BaseFreq={_BaseFreq}')

        _Enabled = fields.get('Enabled')
        if _Enabled is not None:
            output.write(f' Enabled={_Enabled}')

        output.write('\n')

    @model_validator(mode="before")
    def _val_dss_model(self) -> "Fault":
        _fields_set = set(self.keys())
        # Validate oneOf (spec. sets) based on the `required` lists
        _required_r = _fields_set.issuperset({'R'})
        _required_Gmatrix = _fields_set.issuperset({'GMatrix'})
        num_specs = _required_r + _required_Gmatrix
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



Fault_ = Fault


class FaultList(RootModel[List[Fault]]):
    root: List[Fault]





class FaultContainer(RootModel[Union[FaultList, JSONFilePath, JSONLinesFilePath]]):
    root: Union[FaultList, JSONFilePath, JSONLinesFilePath] = Field(..., title="FaultContainer")


    @model_validator(mode="before")
    def _val_dss_model(self) -> "FaultContainer":
        try:
            _fields_set = set(self.keys())
        except:
            if isinstance(self, (list, tuple)):
                return self

            raise

        # Validate oneOf (spec. sets) based on the `required` lists
        _required_FaultList = "root" in _fields_set and isinstance(self.root, FaultList)
        _required_JSONFilePath = _fields_set.issuperset({'JSONFile'})
        _required_JSONLinesFilePath = _fields_set.issuperset({'JSONLinesFile'})
        num_specs = _required_FaultList + _required_JSONFilePath + _required_JSONLinesFilePath
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



class DynamicExp(BaseModel):
    Name: Annotated[str, Field(max_length=255, min_length=1)] = Field(..., title="Name")
    Like: Optional[str] = Field(None, title="Like")
    NVariables: Optional[int] = Field(None, title="NVariables")
    VarNames: Optional[StringArrayOrFilePath] = Field(None, title="VarNames")
    Expression: str = Field(..., title="Expression")
    Domain: Optional[DynamicExpDomain] = Field(None, title="Domain")

    def dumps_dss(self, edit=False) -> str:
        """
        Dump the DSS object dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            self.dump_dss(output, edit)
            return output.getvalue()

    def dump_dss(self, output: TextIO, edit=False):
        """
        Dump the DSS object to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        DynamicExp.dict_dump_dss(self.model_dump(exclude_unset=True), output, edit=edit)

    @staticmethod
    def dict_dumps_dss(fields: Dict, edit=False) -> str:
        """
        Dump the DSS object from a dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            DynamicExp.dict_dump_dss(fields, output, edit)
            return output.getvalue()

    @staticmethod
    def dict_dump_dss(fields: Dict, output: TextIO, edit=False):
        """
        Dump the DSS object from a dict to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        if edit:
            output.write(f'''edit DynamicExp.{fields['Name']}''')
        else:
            output.write(f'''new DynamicExp.{fields['Name']}''')

        _Like = fields.get('Like')
        if _Like is not None:
            output.write(f' Like={_quoted(_Like)}')

        _NVariables = fields.get('NVariables')
        if _NVariables is not None:
            output.write(f' NVariables={_NVariables}')

        _VarNames = fields.get('VarNames')
        if _VarNames is not None:
            output.write(f' VarNames=({_filepath_stringlist(_VarNames)})')

        _Expression = fields.get('Expression')
        output.write(f' Expression={_quoted(_Expression)}')
        _Domain = fields.get('Domain')
        if _Domain is not None:
            output.write(f' Domain={_quoted(_Domain)}')

        output.write('\n')


DynamicExp_ = DynamicExp


class DynamicExpList(RootModel[List[DynamicExp]]):
    root: List[DynamicExp]





class DynamicExpContainer(RootModel[Union[DynamicExpList, JSONFilePath, JSONLinesFilePath]]):
    root: Union[DynamicExpList, JSONFilePath, JSONLinesFilePath] = Field(..., title="DynamicExpContainer")


    @model_validator(mode="before")
    def _val_dss_model(self) -> "DynamicExpContainer":
        try:
            _fields_set = set(self.keys())
        except:
            if isinstance(self, (list, tuple)):
                return self

            raise

        # Validate oneOf (spec. sets) based on the `required` lists
        _required_DynamicExpList = "root" in _fields_set and isinstance(self.root, DynamicExpList)
        _required_JSONFilePath = _fields_set.issuperset({'JSONFile'})
        _required_JSONLinesFilePath = _fields_set.issuperset({'JSONLinesFile'})
        num_specs = _required_DynamicExpList + _required_JSONFilePath + _required_JSONLinesFilePath
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



class Generator_Common(BaseModel):
    Name: Annotated[str, Field(max_length=255, min_length=1)] = Field(..., title="Name")
    Like: Optional[str] = Field(None, title="Like")
    Phases: Optional[PositiveInt] = Field(None, title="Phases")
    Bus1: BusConnection = Field(..., title="Bus1")
    kV: Annotated[float, Field(ge=0)] = Field(..., title="kV")
    Model: Optional[GeneratorModel] = Field(None, title="Model")
    VMinpu: Optional[float] = Field(None, title="VMinpu")
    VMaxpu: Optional[float] = Field(None, title="VMaxpu")
    Yearly: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="Yearly")
    Daily: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="Daily")
    Duty: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="Duty")
    DispMode: Optional[GeneratorDispatchMode] = Field(None, title="DispMode")
    DispValue: Optional[float] = Field(None, title="DispValue")
    Conn: Optional[Connection] = Field(None, title="Conn")
    Status: Optional[GeneratorStatus] = Field(None, title="Status")
    Class: Optional[int] = Field(None, title="Class")
    Vpu: Optional[float] = Field(None, title="Vpu")
    Maxkvar: Optional[float] = Field(None, title="Maxkvar")
    Minkvar: Optional[float] = Field(None, title="Minkvar")
    PVFactor: Optional[float] = Field(None, title="PVFactor")
    ForceOn: Optional[bool] = Field(None, title="ForceOn")
    kVA: Optional[float] = Field(None, title="kVA")
    Xd: Optional[float] = Field(None, title="Xd")
    Xdp: Optional[float] = Field(None, title="Xdp")
    Xdpp: Optional[float] = Field(None, title="Xdpp")
    H: Optional[float] = Field(None, title="H")
    D: Optional[float] = Field(None, title="D")
    UserModel: Optional[FilePath] = Field(None, title="UserModel")
    UserData: Optional[str] = Field(None, title="UserData")
    ShaftModel: Optional[FilePath] = Field(None, title="ShaftModel")
    ShaftData: Optional[str] = Field(None, title="ShaftData")
    DutyStart: Optional[float] = Field(None, title="DutyStart")
    DebugTrace: Optional[bool] = Field(None, title="DebugTrace")
    Balanced: Optional[bool] = Field(None, title="Balanced")
    XRdp: Optional[float] = Field(None, title="XRdp")
    UseFuel: Optional[bool] = Field(None, title="UseFuel")
    FuelkWh: Optional[float] = Field(None, title="FuelkWh")
    pctFuel: Optional[float] = Field(None, title="%Fuel", validation_alias=AliasChoices("pctFuel", "%Fuel"))
    pctReserve: Optional[float] = Field(None, title="%Reserve", validation_alias=AliasChoices("pctReserve", "%Reserve"))
    DynamicEq: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="DynamicEq")
    DynOut: Optional[StringArrayOrFilePath] = Field(None, title="DynOut")
    Spectrum: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="Spectrum")
    BaseFreq: Optional[PositiveFloat] = Field(None, title="BaseFreq")
    Enabled: Optional[bool] = Field(None, title="Enabled")
    Refuel: Optional[bool] = Field(None, title="Refuel")
    DynInit: Optional[DynInitType] = Field(None, title="DynInit")

class Generator_kWpf(Generator_Common):
    kW: float = Field(..., title="kW")
    PF: Annotated[float, Field(le=1, ge=-1)] = Field(..., title="PF")

class Generator_kWkvar(Generator_Common):
    kW: float = Field(..., title="kW")
    kvar: float = Field(..., title="kvar")


class Generator(RootModel[Union[Generator_kWpf, Generator_kWkvar]]):
    root: Union[Generator_kWpf, Generator_kWkvar] = Field(..., title="Generator")

    def dumps_dss(self, edit=False) -> str:
        """
        Dump the DSS object dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            self.dump_dss(output, edit)
            return output.getvalue()

    def dump_dss(self, output: TextIO, edit=False):
        """
        Dump the DSS object to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        Generator.dict_dump_dss(self.model_dump(exclude_unset=True), output, edit=edit)

    @staticmethod
    def dict_dumps_dss(fields: Dict, edit=False) -> str:
        """
        Dump the DSS object from a dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            Generator.dict_dump_dss(fields, output, edit)
            return output.getvalue()

    @staticmethod
    def dict_dump_dss(fields: Dict, output: TextIO, edit=False):
        """
        Dump the DSS object from a dict to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        if edit:
            output.write(f'''edit Generator.{fields['Name']}''')
        else:
            output.write(f'''new Generator.{fields['Name']}''')

        _Like = fields.get('Like')
        if _Like is not None:
            output.write(f' Like={_quoted(_Like)}')

        _Phases = fields.get('Phases')
        if _Phases is not None:
            output.write(f' Phases={_Phases}')

        _Bus1 = fields.get('Bus1')
        output.write(f' Bus1={_quoted(_Bus1)}')
        _kV = fields.get('kV')
        output.write(f' kV={_kV}')
        _kW = fields.get('kW')
        if _kW is not None:
            output.write(f' kW={_kW}')

        _PF = fields.get('PF')
        if _PF is not None:
            output.write(f' PF={_PF}')

        _kvar = fields.get('kvar')
        if _kvar is not None:
            output.write(f' kvar={_kvar}')

        _Model = fields.get('Model')
        if _Model is not None:
            output.write(f' Model={_Model}')

        _VMinpu = fields.get('VMinpu')
        if _VMinpu is not None:
            output.write(f' VMinpu={_VMinpu}')

        _VMaxpu = fields.get('VMaxpu')
        if _VMaxpu is not None:
            output.write(f' VMaxpu={_VMaxpu}')

        _Yearly = fields.get('Yearly')
        if _Yearly is not None:
            output.write(f' Yearly={_quoted(_Yearly)}')

        _Daily = fields.get('Daily')
        if _Daily is not None:
            output.write(f' Daily={_quoted(_Daily)}')

        _Duty = fields.get('Duty')
        if _Duty is not None:
            output.write(f' Duty={_quoted(_Duty)}')

        _DispMode = fields.get('DispMode')
        if _DispMode is not None:
            output.write(f' DispMode={_quoted(_DispMode)}')

        _DispValue = fields.get('DispValue')
        if _DispValue is not None:
            output.write(f' DispValue={_DispValue}')

        _Conn = fields.get('Conn')
        if _Conn is not None:
            output.write(f' Conn={_quoted(_Conn)}')

        _Status = fields.get('Status')
        if _Status is not None:
            output.write(f' Status={_quoted(_Status)}')

        _Class = fields.get('Class')
        if _Class is not None:
            output.write(f' Class={_Class}')

        _Vpu = fields.get('Vpu')
        if _Vpu is not None:
            output.write(f' Vpu={_Vpu}')

        _Maxkvar = fields.get('Maxkvar')
        if _Maxkvar is not None:
            output.write(f' Maxkvar={_Maxkvar}')

        _Minkvar = fields.get('Minkvar')
        if _Minkvar is not None:
            output.write(f' Minkvar={_Minkvar}')

        _PVFactor = fields.get('PVFactor')
        if _PVFactor is not None:
            output.write(f' PVFactor={_PVFactor}')

        _ForceOn = fields.get('ForceOn')
        if _ForceOn is not None:
            output.write(f' ForceOn={_ForceOn}')

        _kVA = fields.get('kVA')
        if _kVA is not None:
            output.write(f' kVA={_kVA}')

        _Xd = fields.get('Xd')
        if _Xd is not None:
            output.write(f' Xd={_Xd}')

        _Xdp = fields.get('Xdp')
        if _Xdp is not None:
            output.write(f' Xdp={_Xdp}')

        _Xdpp = fields.get('Xdpp')
        if _Xdpp is not None:
            output.write(f' Xdpp={_Xdpp}')

        _H = fields.get('H')
        if _H is not None:
            output.write(f' H={_H}')

        _D = fields.get('D')
        if _D is not None:
            output.write(f' D={_D}')

        _UserModel = fields.get('UserModel')
        if _UserModel is not None:
            output.write(f' UserModel={_quoted(_UserModel)}')

        _UserData = fields.get('UserData')
        if _UserData is not None:
            output.write(f' UserData={_quoted(_UserData)}')

        _ShaftModel = fields.get('ShaftModel')
        if _ShaftModel is not None:
            output.write(f' ShaftModel={_quoted(_ShaftModel)}')

        _ShaftData = fields.get('ShaftData')
        if _ShaftData is not None:
            output.write(f' ShaftData={_quoted(_ShaftData)}')

        _DutyStart = fields.get('DutyStart')
        if _DutyStart is not None:
            output.write(f' DutyStart={_DutyStart}')

        _DebugTrace = fields.get('DebugTrace')
        if _DebugTrace is not None:
            output.write(f' DebugTrace={_DebugTrace}')

        _Balanced = fields.get('Balanced')
        if _Balanced is not None:
            output.write(f' Balanced={_Balanced}')

        _XRdp = fields.get('XRdp')
        if _XRdp is not None:
            output.write(f' XRdp={_XRdp}')

        _UseFuel = fields.get('UseFuel')
        if _UseFuel is not None:
            output.write(f' UseFuel={_UseFuel}')

        _FuelkWh = fields.get('FuelkWh')
        if _FuelkWh is not None:
            output.write(f' FuelkWh={_FuelkWh}')

        _pctFuel = fields.get('pctFuel')
        if _pctFuel is not None:
            output.write(f' %Fuel={_pctFuel}')

        _pctReserve = fields.get('pctReserve')
        if _pctReserve is not None:
            output.write(f' %Reserve={_pctReserve}')

        _DynamicEq = fields.get('DynamicEq')
        if _DynamicEq is not None:
            output.write(f' DynamicEq={_quoted(_DynamicEq)}')

        _DynOut = fields.get('DynOut')
        if _DynOut is not None:
            output.write(f' DynOut=({_filepath_stringlist(_DynOut)})')

        _Spectrum = fields.get('Spectrum')
        if _Spectrum is not None:
            output.write(f' Spectrum={_quoted(_Spectrum)}')

        _BaseFreq = fields.get('BaseFreq')
        if _BaseFreq is not None:
            output.write(f' BaseFreq={_BaseFreq}')

        _Enabled = fields.get('Enabled')
        if _Enabled is not None:
            output.write(f' Enabled={_Enabled}')

        _Refuel = fields.get('Refuel')
        if _Refuel is not None:
            output.write(f' Refuel={_Refuel}')

        _DynInit = fields.get('DynInit')
        if _DynInit is not None:
            DynInitType.dict_dump_dss(_DynInit, output)

        output.write('\n')

    @model_validator(mode="before")
    def _val_dss_model(self) -> "Generator":
        _fields_set = set(self.keys())
        # Validate oneOf (spec. sets) based on the `required` lists
        _required_kWpf = _fields_set.issuperset({'kW', 'PF'})
        _required_kWkvar = _fields_set.issuperset({'kvar', 'kW'})
        num_specs = _required_kWpf + _required_kWkvar
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



Generator_ = Generator


class GeneratorList(RootModel[List[Generator]]):
    root: List[Generator]





class GeneratorContainer(RootModel[Union[GeneratorList, JSONFilePath, JSONLinesFilePath]]):
    root: Union[GeneratorList, JSONFilePath, JSONLinesFilePath] = Field(..., title="GeneratorContainer")


    @model_validator(mode="before")
    def _val_dss_model(self) -> "GeneratorContainer":
        try:
            _fields_set = set(self.keys())
        except:
            if isinstance(self, (list, tuple)):
                return self

            raise

        # Validate oneOf (spec. sets) based on the `required` lists
        _required_GeneratorList = "root" in _fields_set and isinstance(self.root, GeneratorList)
        _required_JSONFilePath = _fields_set.issuperset({'JSONFile'})
        _required_JSONLinesFilePath = _fields_set.issuperset({'JSONLinesFile'})
        num_specs = _required_GeneratorList + _required_JSONFilePath + _required_JSONLinesFilePath
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



class GenDispatcher(BaseModel):
    Name: Annotated[str, Field(max_length=255, min_length=1)] = Field(..., title="Name")
    Like: Optional[str] = Field(None, title="Like")
    Element: str = Field(..., title="Element")
    Terminal: Optional[int] = Field(None, title="Terminal")
    kWLimit: Optional[float] = Field(None, title="kWLimit")
    kWBand: Optional[float] = Field(None, title="kWBand")
    kvarLimit: Optional[float] = Field(None, title="kvarLimit")
    GenList: Optional[StringArrayOrFilePath] = Field(None, title="GenList")
    Weights: Optional[ArrayOrFilePath] = Field(None, title="Weights")
    BaseFreq: Optional[PositiveFloat] = Field(None, title="BaseFreq")
    Enabled: Optional[bool] = Field(None, title="Enabled")

    def dumps_dss(self, edit=False) -> str:
        """
        Dump the DSS object dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            self.dump_dss(output, edit)
            return output.getvalue()

    def dump_dss(self, output: TextIO, edit=False):
        """
        Dump the DSS object to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        GenDispatcher.dict_dump_dss(self.model_dump(exclude_unset=True), output, edit=edit)

    @staticmethod
    def dict_dumps_dss(fields: Dict, edit=False) -> str:
        """
        Dump the DSS object from a dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            GenDispatcher.dict_dump_dss(fields, output, edit)
            return output.getvalue()

    @staticmethod
    def dict_dump_dss(fields: Dict, output: TextIO, edit=False):
        """
        Dump the DSS object from a dict to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        if edit:
            output.write(f'''edit GenDispatcher.{fields['Name']}''')
        else:
            output.write(f'''new GenDispatcher.{fields['Name']}''')

        # NOTE: "GenList" is redundant, left for clarity

        _Like = fields.get('Like')
        if _Like is not None:
            output.write(f' Like={_quoted(_Like)}')

        _Element = fields.get('Element')
        output.write(f' Element={_quoted(_Element)}')
        _Terminal = fields.get('Terminal')
        if _Terminal is not None:
            output.write(f' Terminal={_Terminal}')

        _kWLimit = fields.get('kWLimit')
        if _kWLimit is not None:
            output.write(f' kWLimit={_kWLimit}')

        _kWBand = fields.get('kWBand')
        if _kWBand is not None:
            output.write(f' kWBand={_kWBand}')

        _kvarLimit = fields.get('kvarLimit')
        if _kvarLimit is not None:
            output.write(f' kvarLimit={_kvarLimit}')

        _GenList = fields.get('GenList')
        if _GenList is not None:
            _length_GenList, _GenList = _filepath_stringlist(_GenList, length=True)
            output.write(f' GenList=({_GenList})')

        _Weights = fields.get('Weights')
        if _Weights is not None:
            if isinstance(_Weights, ARRAY_LIKE):
                if len(_Weights) != _length_GenList:
                    raise ValueError(f'Array length ({len(_Weights)}) for "Weights" does not match expected length ({_length_GenList})')

                _Weights = _as_list(_Weights)
            else:
                _length_Weights, _Weights = _filepath_array(_Weights)
                if _length_GenList != _length_Weights:
                    raise ValueError(f'Array length ({_length_Weights}) for "Weights" (from file) does not match expected length ({_length_GenList})')

            output.write(f' Weights={_Weights}')

        _BaseFreq = fields.get('BaseFreq')
        if _BaseFreq is not None:
            output.write(f' BaseFreq={_BaseFreq}')

        _Enabled = fields.get('Enabled')
        if _Enabled is not None:
            output.write(f' Enabled={_Enabled}')

        output.write('\n')


GenDispatcher_ = GenDispatcher


class GenDispatcherList(RootModel[List[GenDispatcher]]):
    root: List[GenDispatcher]





class GenDispatcherContainer(RootModel[Union[GenDispatcherList, JSONFilePath, JSONLinesFilePath]]):
    root: Union[GenDispatcherList, JSONFilePath, JSONLinesFilePath] = Field(..., title="GenDispatcherContainer")


    @model_validator(mode="before")
    def _val_dss_model(self) -> "GenDispatcherContainer":
        try:
            _fields_set = set(self.keys())
        except:
            if isinstance(self, (list, tuple)):
                return self

            raise

        # Validate oneOf (spec. sets) based on the `required` lists
        _required_GenDispatcherList = "root" in _fields_set and isinstance(self.root, GenDispatcherList)
        _required_JSONFilePath = _fields_set.issuperset({'JSONFile'})
        _required_JSONLinesFilePath = _fields_set.issuperset({'JSONLinesFile'})
        num_specs = _required_GenDispatcherList + _required_JSONFilePath + _required_JSONLinesFilePath
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



class Storage_Common(BaseModel):
    Name: Annotated[str, Field(max_length=255, min_length=1)] = Field(..., title="Name")
    Like: Optional[str] = Field(None, title="Like")
    Phases: Optional[PositiveInt] = Field(None, title="Phases")
    Bus1: BusConnection = Field(..., title="Bus1")
    kV: Annotated[float, Field(ge=0)] = Field(..., title="kV")
    Conn: Optional[Connection] = Field(None, title="Conn")
    kW: Optional[float] = Field(None, title="kW")
    kVA: Optional[float] = Field(None, title="kVA")
    pctCutIn: Optional[float] = Field(None, title="%CutIn", validation_alias=AliasChoices("pctCutIn", "%CutIn"))
    pctCutOut: Optional[float] = Field(None, title="%CutOut", validation_alias=AliasChoices("pctCutOut", "%CutOut"))
    EffCurve: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="EffCurve")
    VarFollowInverter: Optional[bool] = Field(None, title="VarFollowInverter")
    kvarMax: Optional[float] = Field(None, title="kvarMax")
    kvarMaxAbs: Optional[float] = Field(None, title="kvarMaxAbs")
    WattPriority: Optional[bool] = Field(None, title="WattPriority")
    PFPriority: Optional[bool] = Field(None, title="PFPriority")
    pctPMinNoVars: Optional[float] = Field(None, title="%PMinNoVars", validation_alias=AliasChoices("pctPMinNoVars", "%PMinNoVars"))
    pctPMinkvarMax: Optional[float] = Field(None, title="%PMinkvarMax", validation_alias=AliasChoices("pctPMinkvarMax", "%PMinkvarMax"))
    pctkWRated: Optional[float] = Field(None, title="%kWRated", validation_alias=AliasChoices("pctkWRated", "%kWRated"))
    kWhRated: Optional[float] = Field(None, title="kWhRated")
    kWhStored: Optional[Annotated[float, Field(ge=0)]] = Field(None, title="kWhStored")
    pctStored: Optional[float] = Field(None, title="%Stored", validation_alias=AliasChoices("pctStored", "%Stored"))
    pctReserve: Optional[float] = Field(None, title="%Reserve", validation_alias=AliasChoices("pctReserve", "%Reserve"))
    State: Optional[StorageState] = Field(None, title="State")
    pctDischarge: Optional[float] = Field(None, title="%Discharge", validation_alias=AliasChoices("pctDischarge", "%Discharge"))
    pctCharge: Optional[float] = Field(None, title="%Charge", validation_alias=AliasChoices("pctCharge", "%Charge"))
    pctEffCharge: Optional[float] = Field(None, title="%EffCharge", validation_alias=AliasChoices("pctEffCharge", "%EffCharge"))
    pctEffDischarge: Optional[float] = Field(None, title="%EffDischarge", validation_alias=AliasChoices("pctEffDischarge", "%EffDischarge"))
    pctIdlingkW: Optional[float] = Field(None, title="%IdlingkW", validation_alias=AliasChoices("pctIdlingkW", "%IdlingkW"))
    pctR: Optional[float] = Field(None, title="%R", validation_alias=AliasChoices("pctR", "%R"))
    pctX: Optional[float] = Field(None, title="%X", validation_alias=AliasChoices("pctX", "%X"))
    Model: Optional[int] = Field(None, title="Model")
    VMinpu: Optional[float] = Field(None, title="VMinpu")
    VMaxpu: Optional[float] = Field(None, title="VMaxpu")
    Balanced: Optional[bool] = Field(None, title="Balanced")
    LimitCurrent: Optional[bool] = Field(None, title="LimitCurrent")
    Yearly: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="Yearly")
    Daily: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="Daily")
    Duty: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="Duty")
    DispMode: Optional[StorageDispatchMode] = Field(None, title="DispMode")
    DischargeTrigger: Optional[float] = Field(None, title="DischargeTrigger")
    ChargeTrigger: Optional[float] = Field(None, title="ChargeTrigger")
    TimeChargeTrig: Optional[Annotated[float, Field(lt=24, ge=0)]] = Field(None, title="TimeChargeTrig")
    Class: Optional[int] = Field(None, title="Class")
    DynaDLL: Optional[FilePath] = Field(None, title="DynaDLL")
    DynaData: Optional[str] = Field(None, title="DynaData")
    UserModel: Optional[FilePath] = Field(None, title="UserModel")
    UserData: Optional[str] = Field(None, title="UserData")
    DebugTrace: Optional[bool] = Field(None, title="DebugTrace")
    kVDC: Optional[float] = Field(None, title="kVDC")
    Kp: Optional[float] = Field(None, title="Kp")
    PITol: Optional[float] = Field(None, title="PITol")
    SafeVoltage: Optional[float] = Field(None, title="SafeVoltage")
    DynamicEq: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="DynamicEq")
    DynOut: Optional[StringArrayOrFilePath] = Field(None, title="DynOut")
    ControlMode: Optional[InverterControlMode] = Field(None, title="ControlMode")
    AmpLimit: Optional[float] = Field(None, title="AmpLimit")
    AmpLimitGain: Optional[float] = Field(None, title="AmpLimitGain")
    Spectrum: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="Spectrum")
    BaseFreq: Optional[PositiveFloat] = Field(None, title="BaseFreq")
    Enabled: Optional[bool] = Field(None, title="Enabled")
    DynInit: Optional[DynInitType] = Field(None, title="DynInit")

class Storage_kWRatedPF(Storage_Common):
    PF: Annotated[float, Field(le=1, ge=-1)] = Field(..., title="PF")
    kWRated: float = Field(..., title="kWRated")

class Storage_kWRatedkvar(Storage_Common):
    kvar: float = Field(..., title="kvar")
    kWRated: float = Field(..., title="kWRated")


class Storage(RootModel[Union[Storage_kWRatedPF, Storage_kWRatedkvar]]):
    root: Union[Storage_kWRatedPF, Storage_kWRatedkvar] = Field(..., title="Storage")

    def dumps_dss(self, edit=False) -> str:
        """
        Dump the DSS object dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            self.dump_dss(output, edit)
            return output.getvalue()

    def dump_dss(self, output: TextIO, edit=False):
        """
        Dump the DSS object to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        Storage.dict_dump_dss(self.model_dump(exclude_unset=True), output, edit=edit)

    @staticmethod
    def dict_dumps_dss(fields: Dict, edit=False) -> str:
        """
        Dump the DSS object from a dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            Storage.dict_dump_dss(fields, output, edit)
            return output.getvalue()

    @staticmethod
    def dict_dump_dss(fields: Dict, output: TextIO, edit=False):
        """
        Dump the DSS object from a dict to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        if edit:
            output.write(f'''edit Storage.{fields['Name']}''')
        else:
            output.write(f'''new Storage.{fields['Name']}''')

        _Like = fields.get('Like')
        if _Like is not None:
            output.write(f' Like={_quoted(_Like)}')

        _Phases = fields.get('Phases')
        if _Phases is not None:
            output.write(f' Phases={_Phases}')

        _Bus1 = fields.get('Bus1')
        output.write(f' Bus1={_quoted(_Bus1)}')
        _kV = fields.get('kV')
        output.write(f' kV={_kV}')
        _Conn = fields.get('Conn')
        if _Conn is not None:
            output.write(f' Conn={_quoted(_Conn)}')

        _kW = fields.get('kW')
        if _kW is not None:
            output.write(f' kW={_kW}')

        _kvar = fields.get('kvar')
        if _kvar is not None:
            output.write(f' kvar={_kvar}')

        _PF = fields.get('PF')
        if _PF is not None:
            output.write(f' PF={_PF}')

        _kVA = fields.get('kVA')
        if _kVA is not None:
            output.write(f' kVA={_kVA}')

        _pctCutIn = fields.get('pctCutIn')
        if _pctCutIn is not None:
            output.write(f' %CutIn={_pctCutIn}')

        _pctCutOut = fields.get('pctCutOut')
        if _pctCutOut is not None:
            output.write(f' %CutOut={_pctCutOut}')

        _EffCurve = fields.get('EffCurve')
        if _EffCurve is not None:
            output.write(f' EffCurve={_quoted(_EffCurve)}')

        _VarFollowInverter = fields.get('VarFollowInverter')
        if _VarFollowInverter is not None:
            output.write(f' VarFollowInverter={_VarFollowInverter}')

        _kvarMax = fields.get('kvarMax')
        if _kvarMax is not None:
            output.write(f' kvarMax={_kvarMax}')

        _kvarMaxAbs = fields.get('kvarMaxAbs')
        if _kvarMaxAbs is not None:
            output.write(f' kvarMaxAbs={_kvarMaxAbs}')

        _WattPriority = fields.get('WattPriority')
        if _WattPriority is not None:
            output.write(f' WattPriority={_WattPriority}')

        _PFPriority = fields.get('PFPriority')
        if _PFPriority is not None:
            output.write(f' PFPriority={_PFPriority}')

        _pctPMinNoVars = fields.get('pctPMinNoVars')
        if _pctPMinNoVars is not None:
            output.write(f' %PMinNoVars={_pctPMinNoVars}')

        _pctPMinkvarMax = fields.get('pctPMinkvarMax')
        if _pctPMinkvarMax is not None:
            output.write(f' %PMinkvarMax={_pctPMinkvarMax}')

        _kWRated = fields.get('kWRated')
        if _kWRated is not None:
            output.write(f' kWRated={_kWRated}')

        _pctkWRated = fields.get('pctkWRated')
        if _pctkWRated is not None:
            output.write(f' %kWRated={_pctkWRated}')

        _kWhRated = fields.get('kWhRated')
        if _kWhRated is not None:
            output.write(f' kWhRated={_kWhRated}')

        _kWhStored = fields.get('kWhStored')
        if _kWhStored is not None:
            output.write(f' kWhStored={_kWhStored}')

        _pctStored = fields.get('pctStored')
        if _pctStored is not None:
            output.write(f' %Stored={_pctStored}')

        _pctReserve = fields.get('pctReserve')
        if _pctReserve is not None:
            output.write(f' %Reserve={_pctReserve}')

        _State = fields.get('State')
        if _State is not None:
            output.write(f' State={_quoted(_State)}')

        _pctDischarge = fields.get('pctDischarge')
        if _pctDischarge is not None:
            output.write(f' %Discharge={_pctDischarge}')

        _pctCharge = fields.get('pctCharge')
        if _pctCharge is not None:
            output.write(f' %Charge={_pctCharge}')

        _pctEffCharge = fields.get('pctEffCharge')
        if _pctEffCharge is not None:
            output.write(f' %EffCharge={_pctEffCharge}')

        _pctEffDischarge = fields.get('pctEffDischarge')
        if _pctEffDischarge is not None:
            output.write(f' %EffDischarge={_pctEffDischarge}')

        _pctIdlingkW = fields.get('pctIdlingkW')
        if _pctIdlingkW is not None:
            output.write(f' %IdlingkW={_pctIdlingkW}')

        _pctR = fields.get('pctR')
        if _pctR is not None:
            output.write(f' %R={_pctR}')

        _pctX = fields.get('pctX')
        if _pctX is not None:
            output.write(f' %X={_pctX}')

        _Model = fields.get('Model')
        if _Model is not None:
            output.write(f' Model={_Model}')

        _VMinpu = fields.get('VMinpu')
        if _VMinpu is not None:
            output.write(f' VMinpu={_VMinpu}')

        _VMaxpu = fields.get('VMaxpu')
        if _VMaxpu is not None:
            output.write(f' VMaxpu={_VMaxpu}')

        _Balanced = fields.get('Balanced')
        if _Balanced is not None:
            output.write(f' Balanced={_Balanced}')

        _LimitCurrent = fields.get('LimitCurrent')
        if _LimitCurrent is not None:
            output.write(f' LimitCurrent={_LimitCurrent}')

        _Yearly = fields.get('Yearly')
        if _Yearly is not None:
            output.write(f' Yearly={_quoted(_Yearly)}')

        _Daily = fields.get('Daily')
        if _Daily is not None:
            output.write(f' Daily={_quoted(_Daily)}')

        _Duty = fields.get('Duty')
        if _Duty is not None:
            output.write(f' Duty={_quoted(_Duty)}')

        _DispMode = fields.get('DispMode')
        if _DispMode is not None:
            output.write(f' DispMode={_quoted(_DispMode)}')

        _DischargeTrigger = fields.get('DischargeTrigger')
        if _DischargeTrigger is not None:
            output.write(f' DischargeTrigger={_DischargeTrigger}')

        _ChargeTrigger = fields.get('ChargeTrigger')
        if _ChargeTrigger is not None:
            output.write(f' ChargeTrigger={_ChargeTrigger}')

        _TimeChargeTrig = fields.get('TimeChargeTrig')
        if _TimeChargeTrig is not None:
            output.write(f' TimeChargeTrig={_TimeChargeTrig}')

        _Class = fields.get('Class')
        if _Class is not None:
            output.write(f' Class={_Class}')

        _DynaDLL = fields.get('DynaDLL')
        if _DynaDLL is not None:
            output.write(f' DynaDLL={_quoted(_DynaDLL)}')

        _DynaData = fields.get('DynaData')
        if _DynaData is not None:
            output.write(f' DynaData={_quoted(_DynaData)}')

        _UserModel = fields.get('UserModel')
        if _UserModel is not None:
            output.write(f' UserModel={_quoted(_UserModel)}')

        _UserData = fields.get('UserData')
        if _UserData is not None:
            output.write(f' UserData={_quoted(_UserData)}')

        _DebugTrace = fields.get('DebugTrace')
        if _DebugTrace is not None:
            output.write(f' DebugTrace={_DebugTrace}')

        _kVDC = fields.get('kVDC')
        if _kVDC is not None:
            output.write(f' kVDC={_kVDC}')

        _Kp = fields.get('Kp')
        if _Kp is not None:
            output.write(f' Kp={_Kp}')

        _PITol = fields.get('PITol')
        if _PITol is not None:
            output.write(f' PITol={_PITol}')

        _SafeVoltage = fields.get('SafeVoltage')
        if _SafeVoltage is not None:
            output.write(f' SafeVoltage={_SafeVoltage}')

        _DynamicEq = fields.get('DynamicEq')
        if _DynamicEq is not None:
            output.write(f' DynamicEq={_quoted(_DynamicEq)}')

        _DynOut = fields.get('DynOut')
        if _DynOut is not None:
            output.write(f' DynOut=({_filepath_stringlist(_DynOut)})')

        _ControlMode = fields.get('ControlMode')
        if _ControlMode is not None:
            output.write(f' ControlMode={_quoted(_ControlMode)}')

        _AmpLimit = fields.get('AmpLimit')
        if _AmpLimit is not None:
            output.write(f' AmpLimit={_AmpLimit}')

        _AmpLimitGain = fields.get('AmpLimitGain')
        if _AmpLimitGain is not None:
            output.write(f' AmpLimitGain={_AmpLimitGain}')

        _Spectrum = fields.get('Spectrum')
        if _Spectrum is not None:
            output.write(f' Spectrum={_quoted(_Spectrum)}')

        _BaseFreq = fields.get('BaseFreq')
        if _BaseFreq is not None:
            output.write(f' BaseFreq={_BaseFreq}')

        _Enabled = fields.get('Enabled')
        if _Enabled is not None:
            output.write(f' Enabled={_Enabled}')

        _DynInit = fields.get('DynInit')
        if _DynInit is not None:
            DynInitType.dict_dump_dss(_DynInit, output)

        output.write('\n')

    @model_validator(mode="before")
    def _val_dss_model(self) -> "Storage":
        _fields_set = set(self.keys())
        # Validate oneOf (spec. sets) based on the `required` lists
        _required_kWRatedPF = _fields_set.issuperset({'kWRated', 'PF'})
        _required_kWRatedkvar = _fields_set.issuperset({'kvar', 'kWRated'})
        num_specs = _required_kWRatedPF + _required_kWRatedkvar
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



Storage_ = Storage


class StorageList(RootModel[List[Storage]]):
    root: List[Storage]



StorageList_ = StorageList




class StorageContainer(RootModel[Union[StorageList, JSONFilePath, JSONLinesFilePath]]):
    root: Union[StorageList, JSONFilePath, JSONLinesFilePath] = Field(..., title="StorageContainer")


    @model_validator(mode="before")
    def _val_dss_model(self) -> "StorageContainer":
        try:
            _fields_set = set(self.keys())
        except:
            if isinstance(self, (list, tuple)):
                return self

            raise

        # Validate oneOf (spec. sets) based on the `required` lists
        _required_StorageList = "root" in _fields_set and isinstance(self.root, StorageList)
        _required_JSONFilePath = _fields_set.issuperset({'JSONFile'})
        _required_JSONLinesFilePath = _fields_set.issuperset({'JSONLinesFile'})
        num_specs = _required_StorageList + _required_JSONFilePath + _required_JSONLinesFilePath
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



class StorageController(BaseModel):
    Name: Annotated[str, Field(max_length=255, min_length=1)] = Field(..., title="Name")
    Like: Optional[str] = Field(None, title="Like")
    Element: str = Field(..., title="Element")
    Terminal: Optional[int] = Field(None, title="Terminal")
    MonPhase: Optional[MonitoredPhase] = Field(None, title="MonPhase")
    kWTarget: Optional[float] = Field(None, title="kWTarget")
    kWTargetLow: Optional[float] = Field(None, title="kWTargetLow")
    pctkWBand: Optional[float] = Field(None, title="%kWBand", validation_alias=AliasChoices("pctkWBand", "%kWBand"))
    pctkWBandLow: Optional[float] = Field(None, title="%kWBandLow", validation_alias=AliasChoices("pctkWBandLow", "%kWBandLow"))
    ElementList: Optional[StringArrayOrFilePath] = Field(None, title="ElementList")
    Weights: Optional[ArrayOrFilePath] = Field(None, title="Weights")
    ModeDischarge: Optional[StorageControllerDischargeMode] = Field(None, title="ModeDischarge")
    ModeCharge: Optional[StorageControllerChargeMode] = Field(None, title="ModeCharge")
    TimeDischargeTrigger: Optional[float] = Field(None, title="TimeDischargeTrigger")
    TimeChargeTrigger: Optional[float] = Field(None, title="TimeChargeTrigger")
    pctRatekW: Optional[float] = Field(None, title="%RatekW", validation_alias=AliasChoices("pctRatekW", "%RatekW"))
    pctRateCharge: Optional[float] = Field(None, title="%RateCharge", validation_alias=AliasChoices("pctRateCharge", "%RateCharge"))
    pctReserve: Optional[float] = Field(None, title="%Reserve", validation_alias=AliasChoices("pctReserve", "%Reserve"))
    Yearly: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="Yearly")
    Daily: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="Daily")
    Duty: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="Duty")
    EventLog: Optional[bool] = Field(None, title="EventLog")
    InhibitTime: Optional[Annotated[int, Field(ge=0)]] = Field(None, title="InhibitTime")
    TUp: Optional[Annotated[float, Field(ge=0)]] = Field(None, title="TUp")
    TFlat: Optional[Annotated[float, Field(ge=0)]] = Field(None, title="TFlat")
    TDn: Optional[Annotated[float, Field(ge=0)]] = Field(None, title="TDn")
    kWThreshold: Optional[float] = Field(None, title="kWThreshold")
    DispFactor: Optional[float] = Field(None, title="DispFactor")
    ResetLevel: Optional[float] = Field(None, title="ResetLevel")
    SeasonTargets: Optional[ArrayOrFilePath] = Field(None, title="SeasonTargets")
    SeasonTargetsLow: Optional[ArrayOrFilePath] = Field(None, title="SeasonTargetsLow")
    BaseFreq: Optional[PositiveFloat] = Field(None, title="BaseFreq")
    Enabled: Optional[bool] = Field(None, title="Enabled")

    def dumps_dss(self, edit=False) -> str:
        """
        Dump the DSS object dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            self.dump_dss(output, edit)
            return output.getvalue()

    def dump_dss(self, output: TextIO, edit=False):
        """
        Dump the DSS object to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        StorageController.dict_dump_dss(self.model_dump(exclude_unset=True), output, edit=edit)

    @staticmethod
    def dict_dumps_dss(fields: Dict, edit=False) -> str:
        """
        Dump the DSS object from a dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            StorageController.dict_dump_dss(fields, output, edit)
            return output.getvalue()

    @staticmethod
    def dict_dump_dss(fields: Dict, output: TextIO, edit=False):
        """
        Dump the DSS object from a dict to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        _length_Seasons = None

        if edit:
            output.write(f'''edit StorageController.{fields['Name']}''')
        else:
            output.write(f'''new StorageController.{fields['Name']}''')

        # NOTE: "ElementList" is redundant, left for clarity

        _Like = fields.get('Like')
        if _Like is not None:
            output.write(f' Like={_quoted(_Like)}')

        _Element = fields.get('Element')
        output.write(f' Element={_quoted(_Element)}')
        _Terminal = fields.get('Terminal')
        if _Terminal is not None:
            output.write(f' Terminal={_Terminal}')

        _MonPhase = fields.get('MonPhase')
        if _MonPhase is not None:
            output.write(f' MonPhase={_MonPhase}')

        _kWTarget = fields.get('kWTarget')
        if _kWTarget is not None:
            output.write(f' kWTarget={_kWTarget}')

        _kWTargetLow = fields.get('kWTargetLow')
        if _kWTargetLow is not None:
            output.write(f' kWTargetLow={_kWTargetLow}')

        _pctkWBand = fields.get('pctkWBand')
        if _pctkWBand is not None:
            output.write(f' %kWBand={_pctkWBand}')

        _pctkWBandLow = fields.get('pctkWBandLow')
        if _pctkWBandLow is not None:
            output.write(f' %kWBandLow={_pctkWBandLow}')

        _ElementList = fields.get('ElementList')
        if _ElementList is not None:
            _length_ElementList, _ElementList = _filepath_stringlist(_ElementList, length=True)
            output.write(f' ElementList=({_ElementList})')

        _Weights = fields.get('Weights')
        if _Weights is not None:
            if isinstance(_Weights, ARRAY_LIKE):
                if len(_Weights) != _length_ElementList:
                    raise ValueError(f'Array length ({len(_Weights)}) for "Weights" does not match expected length ({_length_ElementList})')

                _Weights = _as_list(_Weights)
            else:
                _length_Weights, _Weights = _filepath_array(_Weights)
                if _length_ElementList != _length_Weights:
                    raise ValueError(f'Array length ({_length_Weights}) for "Weights" (from file) does not match expected length ({_length_ElementList})')

            output.write(f' Weights={_Weights}')

        _ModeDischarge = fields.get('ModeDischarge')
        if _ModeDischarge is not None:
            output.write(f' ModeDischarge={_quoted(_ModeDischarge)}')

        _ModeCharge = fields.get('ModeCharge')
        if _ModeCharge is not None:
            output.write(f' ModeCharge={_quoted(_ModeCharge)}')

        _TimeDischargeTrigger = fields.get('TimeDischargeTrigger')
        if _TimeDischargeTrigger is not None:
            output.write(f' TimeDischargeTrigger={_TimeDischargeTrigger}')

        _TimeChargeTrigger = fields.get('TimeChargeTrigger')
        if _TimeChargeTrigger is not None:
            output.write(f' TimeChargeTrigger={_TimeChargeTrigger}')

        _pctRatekW = fields.get('pctRatekW')
        if _pctRatekW is not None:
            output.write(f' %RatekW={_pctRatekW}')

        _pctRateCharge = fields.get('pctRateCharge')
        if _pctRateCharge is not None:
            output.write(f' %RateCharge={_pctRateCharge}')

        _pctReserve = fields.get('pctReserve')
        if _pctReserve is not None:
            output.write(f' %Reserve={_pctReserve}')

        _Yearly = fields.get('Yearly')
        if _Yearly is not None:
            output.write(f' Yearly={_quoted(_Yearly)}')

        _Daily = fields.get('Daily')
        if _Daily is not None:
            output.write(f' Daily={_quoted(_Daily)}')

        _Duty = fields.get('Duty')
        if _Duty is not None:
            output.write(f' Duty={_quoted(_Duty)}')

        _EventLog = fields.get('EventLog')
        if _EventLog is not None:
            output.write(f' EventLog={_EventLog}')

        _InhibitTime = fields.get('InhibitTime')
        if _InhibitTime is not None:
            output.write(f' InhibitTime={_InhibitTime}')

        _TUp = fields.get('TUp')
        if _TUp is not None:
            output.write(f' TUp={_TUp}')

        _TFlat = fields.get('TFlat')
        if _TFlat is not None:
            output.write(f' TFlat={_TFlat}')

        _TDn = fields.get('TDn')
        if _TDn is not None:
            output.write(f' TDn={_TDn}')

        _kWThreshold = fields.get('kWThreshold')
        if _kWThreshold is not None:
            output.write(f' kWThreshold={_kWThreshold}')

        _DispFactor = fields.get('DispFactor')
        if _DispFactor is not None:
            output.write(f' DispFactor={_DispFactor}')

        _ResetLevel = fields.get('ResetLevel')
        if _ResetLevel is not None:
            output.write(f' ResetLevel={_ResetLevel}')

        _SeasonTargets = fields.get('SeasonTargets')
        if _SeasonTargets is not None:
            if isinstance(_SeasonTargets, ARRAY_LIKE):
                _length_Seasons = len(_SeasonTargets)
                output.write(f' Seasons={_length_Seasons}')
                _SeasonTargets = _as_list(_SeasonTargets)
            else:
                _length_SeasonTargets, _SeasonTargets = _filepath_array(_SeasonTargets)
                if _length_Seasons is None:
                    _length_Seasons = _length_SeasonTargets
                    output.write(f' Seasons={_length_Seasons}')
                elif _length_Seasons != _length_SeasonTargets:
                    raise ValueError(f'Array length ({_length_SeasonTargets}) for "SeasonTargets" (from file) does not match expected length ({_length_Seasons})')

            output.write(f' SeasonTargets={_SeasonTargets}')

        _SeasonTargetsLow = fields.get('SeasonTargetsLow')
        if _SeasonTargetsLow is not None:
            if isinstance(_SeasonTargetsLow, ARRAY_LIKE):
                if _length_Seasons is None:
                    _length_Seasons = len(_SeasonTargetsLow)
                    output.write(f' Seasons={_length_Seasons}')
                elif len(_SeasonTargetsLow) != _length_Seasons:
                    raise ValueError(f'Array length ({len(_SeasonTargetsLow)}) for "SeasonTargetsLow" does not match expected length ({_length_Seasons})')

                _SeasonTargetsLow = _as_list(_SeasonTargetsLow)
            else:
                _length_SeasonTargetsLow, _SeasonTargetsLow = _filepath_array(_SeasonTargetsLow)
                if _length_Seasons is None:
                    _length_Seasons = _length_SeasonTargetsLow
                    output.write(f' Seasons={_length_Seasons}')
                elif _length_Seasons != _length_SeasonTargetsLow:
                    raise ValueError(f'Array length ({_length_SeasonTargetsLow}) for "SeasonTargetsLow" (from file) does not match expected length ({_length_Seasons})')

            output.write(f' SeasonTargetsLow={_SeasonTargetsLow}')

        _BaseFreq = fields.get('BaseFreq')
        if _BaseFreq is not None:
            output.write(f' BaseFreq={_BaseFreq}')

        _Enabled = fields.get('Enabled')
        if _Enabled is not None:
            output.write(f' Enabled={_Enabled}')

        output.write('\n')


StorageController_ = StorageController


class StorageControllerList(RootModel[List[StorageController]]):
    root: List[StorageController]





class StorageControllerContainer(RootModel[Union[StorageControllerList, JSONFilePath, JSONLinesFilePath]]):
    root: Union[StorageControllerList, JSONFilePath, JSONLinesFilePath] = Field(..., title="StorageControllerContainer")


    @model_validator(mode="before")
    def _val_dss_model(self) -> "StorageControllerContainer":
        try:
            _fields_set = set(self.keys())
        except:
            if isinstance(self, (list, tuple)):
                return self

            raise

        # Validate oneOf (spec. sets) based on the `required` lists
        _required_StorageControllerList = "root" in _fields_set and isinstance(self.root, StorageControllerList)
        _required_JSONFilePath = _fields_set.issuperset({'JSONFile'})
        _required_JSONLinesFilePath = _fields_set.issuperset({'JSONLinesFile'})
        num_specs = _required_StorageControllerList + _required_JSONFilePath + _required_JSONLinesFilePath
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



class Relay(BaseModel):
    Name: Annotated[str, Field(max_length=255, min_length=1)] = Field(..., title="Name")
    Like: Optional[str] = Field(None, title="Like")
    MonitoredObj: str = Field(..., title="MonitoredObj")
    MonitoredTerm: Optional[int] = Field(None, title="MonitoredTerm")
    SwitchedObj: Optional[str] = Field(None, title="SwitchedObj")
    SwitchedTerm: Optional[int] = Field(None, title="SwitchedTerm")
    Type: Optional[RelayType] = Field(None, title="Type")
    PhaseCurve: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="PhaseCurve")
    GroundCurve: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="GroundCurve")
    PhaseTrip: Optional[float] = Field(None, title="PhaseTrip")
    GroundTrip: Optional[float] = Field(None, title="GroundTrip")
    TDPhase: Optional[float] = Field(None, title="TDPhase")
    TDGround: Optional[float] = Field(None, title="TDGround")
    PhaseInst: Optional[float] = Field(None, title="PhaseInst")
    GroundInst: Optional[float] = Field(None, title="GroundInst")
    Reset: Optional[float] = Field(None, title="Reset")
    Shots: Optional[PositiveInt] = Field(None, title="Shots")
    RecloseIntervals: Optional[List[float]] = Field(None, title="RecloseIntervals")
    Delay: Optional[float] = Field(None, title="Delay")
    OvervoltCurve: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="OvervoltCurve")
    UndervoltCurve: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="UndervoltCurve")
    kVBase: Optional[float] = Field(None, title="kVBase")
    F47pctPickup: Optional[float] = Field(None, title="47%Pickup", validation_alias=AliasChoices("F47pctPickup", "47%Pickup"))
    F46BaseAmps: Optional[float] = Field(None, title="46BaseAmps", validation_alias=AliasChoices("F46BaseAmps", "46BaseAmps"))
    F46pctPickup: Optional[float] = Field(None, title="46%Pickup", validation_alias=AliasChoices("F46pctPickup", "46%Pickup"))
    F46isqt: Optional[float] = Field(None, title="46isqt", validation_alias=AliasChoices("F46isqt", "46isqt"))
    Variable: Optional[str] = Field(None, title="Variable")
    Overtrip: Optional[float] = Field(None, title="Overtrip")
    Undertrip: Optional[float] = Field(None, title="Undertrip")
    BreakerTime: Optional[float] = Field(None, title="BreakerTime")
    Z1Mag: Optional[float] = Field(None, title="Z1Mag")
    Z1Ang: Optional[float] = Field(None, title="Z1Ang")
    Z0Mag: Optional[float] = Field(None, title="Z0Mag")
    Z0Ang: Optional[float] = Field(None, title="Z0Ang")
    MPhase: Optional[float] = Field(None, title="MPhase")
    MGround: Optional[float] = Field(None, title="MGround")
    EventLog: Optional[bool] = Field(None, title="EventLog")
    DebugTrace: Optional[bool] = Field(None, title="DebugTrace")
    DistReverse: Optional[bool] = Field(None, title="DistReverse")
    Normal: Optional[RelayState] = Field(None, title="Normal")
    State: Optional[RelayState] = Field(None, title="State")
    DOC_TiltAngleLow: Optional[float] = Field(None, title="DOC_TiltAngleLow")
    DOC_TiltAngleHigh: Optional[float] = Field(None, title="DOC_TiltAngleHigh")
    DOC_TripSettingLow: Optional[float] = Field(None, title="DOC_TripSettingLow")
    DOC_TripSettingHigh: Optional[float] = Field(None, title="DOC_TripSettingHigh")
    DOC_TripSettingMag: Optional[float] = Field(None, title="DOC_TripSettingMag")
    DOC_DelayInner: Optional[float] = Field(None, title="DOC_DelayInner")
    DOC_PhaseCurveInner: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="DOC_PhaseCurveInner")
    DOC_PhaseTripInner: Optional[float] = Field(None, title="DOC_PhaseTripInner")
    DOC_TDPhaseInner: Optional[float] = Field(None, title="DOC_TDPhaseInner")
    DOC_P1Blocking: Optional[bool] = Field(None, title="DOC_P1Blocking")
    BaseFreq: Optional[PositiveFloat] = Field(None, title="BaseFreq")
    Enabled: Optional[bool] = Field(None, title="Enabled")

    def dumps_dss(self, edit=False) -> str:
        """
        Dump the DSS object dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            self.dump_dss(output, edit)
            return output.getvalue()

    def dump_dss(self, output: TextIO, edit=False):
        """
        Dump the DSS object to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        Relay.dict_dump_dss(self.model_dump(exclude_unset=True), output, edit=edit)

    @staticmethod
    def dict_dumps_dss(fields: Dict, edit=False) -> str:
        """
        Dump the DSS object from a dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            Relay.dict_dump_dss(fields, output, edit)
            return output.getvalue()

    @staticmethod
    def dict_dump_dss(fields: Dict, output: TextIO, edit=False):
        """
        Dump the DSS object from a dict to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        if edit:
            output.write(f'''edit Relay.{fields['Name']}''')
        else:
            output.write(f'''new Relay.{fields['Name']}''')

        # NOTE: "Shots" is redundant, left for clarity

        _Like = fields.get('Like')
        if _Like is not None:
            output.write(f' Like={_quoted(_Like)}')

        _MonitoredObj = fields.get('MonitoredObj')
        output.write(f' MonitoredObj={_quoted(_MonitoredObj)}')
        _MonitoredTerm = fields.get('MonitoredTerm')
        if _MonitoredTerm is not None:
            output.write(f' MonitoredTerm={_MonitoredTerm}')

        _SwitchedObj = fields.get('SwitchedObj')
        if _SwitchedObj is not None:
            output.write(f' SwitchedObj={_quoted(_SwitchedObj)}')

        _SwitchedTerm = fields.get('SwitchedTerm')
        if _SwitchedTerm is not None:
            output.write(f' SwitchedTerm={_SwitchedTerm}')

        _Type = fields.get('Type')
        if _Type is not None:
            output.write(f' Type={_quoted(_Type)}')

        _PhaseCurve = fields.get('PhaseCurve')
        if _PhaseCurve is not None:
            output.write(f' PhaseCurve={_quoted(_PhaseCurve)}')

        _GroundCurve = fields.get('GroundCurve')
        if _GroundCurve is not None:
            output.write(f' GroundCurve={_quoted(_GroundCurve)}')

        _PhaseTrip = fields.get('PhaseTrip')
        if _PhaseTrip is not None:
            output.write(f' PhaseTrip={_PhaseTrip}')

        _GroundTrip = fields.get('GroundTrip')
        if _GroundTrip is not None:
            output.write(f' GroundTrip={_GroundTrip}')

        _TDPhase = fields.get('TDPhase')
        if _TDPhase is not None:
            output.write(f' TDPhase={_TDPhase}')

        _TDGround = fields.get('TDGround')
        if _TDGround is not None:
            output.write(f' TDGround={_TDGround}')

        _PhaseInst = fields.get('PhaseInst')
        if _PhaseInst is not None:
            output.write(f' PhaseInst={_PhaseInst}')

        _GroundInst = fields.get('GroundInst')
        if _GroundInst is not None:
            output.write(f' GroundInst={_GroundInst}')

        _Reset = fields.get('Reset')
        if _Reset is not None:
            output.write(f' Reset={_Reset}')

        _Shots = fields.get('Shots')
        if _Shots is not None:
            _length_Shots = _Shots - 1
            output.write(f' Shots={_Shots}')

        _RecloseIntervals = fields.get('RecloseIntervals')
        if _RecloseIntervals is not None:
            if len(_RecloseIntervals) != _length_Shots:
                raise ValueError(f'Array length ({len(_RecloseIntervals)}) for "RecloseIntervals" does not match expected length ({_length_Shots})')

            output.write(f' RecloseIntervals={_as_list(_RecloseIntervals)}')

        _Delay = fields.get('Delay')
        if _Delay is not None:
            output.write(f' Delay={_Delay}')

        _OvervoltCurve = fields.get('OvervoltCurve')
        if _OvervoltCurve is not None:
            output.write(f' OvervoltCurve={_quoted(_OvervoltCurve)}')

        _UndervoltCurve = fields.get('UndervoltCurve')
        if _UndervoltCurve is not None:
            output.write(f' UndervoltCurve={_quoted(_UndervoltCurve)}')

        _kVBase = fields.get('kVBase')
        if _kVBase is not None:
            output.write(f' kVBase={_kVBase}')

        _F47pctPickup = fields.get('F47pctPickup')
        if _F47pctPickup is not None:
            output.write(f' 47%Pickup={_F47pctPickup}')

        _F46BaseAmps = fields.get('F46BaseAmps')
        if _F46BaseAmps is not None:
            output.write(f' 46BaseAmps={_F46BaseAmps}')

        _F46pctPickup = fields.get('F46pctPickup')
        if _F46pctPickup is not None:
            output.write(f' 46%Pickup={_F46pctPickup}')

        _F46isqt = fields.get('F46isqt')
        if _F46isqt is not None:
            output.write(f' 46isqt={_F46isqt}')

        _Variable = fields.get('Variable')
        if _Variable is not None:
            output.write(f' Variable={_quoted(_Variable)}')

        _Overtrip = fields.get('Overtrip')
        if _Overtrip is not None:
            output.write(f' Overtrip={_Overtrip}')

        _Undertrip = fields.get('Undertrip')
        if _Undertrip is not None:
            output.write(f' Undertrip={_Undertrip}')

        _BreakerTime = fields.get('BreakerTime')
        if _BreakerTime is not None:
            output.write(f' BreakerTime={_BreakerTime}')

        _Z1Mag = fields.get('Z1Mag')
        if _Z1Mag is not None:
            output.write(f' Z1Mag={_Z1Mag}')

        _Z1Ang = fields.get('Z1Ang')
        if _Z1Ang is not None:
            output.write(f' Z1Ang={_Z1Ang}')

        _Z0Mag = fields.get('Z0Mag')
        if _Z0Mag is not None:
            output.write(f' Z0Mag={_Z0Mag}')

        _Z0Ang = fields.get('Z0Ang')
        if _Z0Ang is not None:
            output.write(f' Z0Ang={_Z0Ang}')

        _MPhase = fields.get('MPhase')
        if _MPhase is not None:
            output.write(f' MPhase={_MPhase}')

        _MGround = fields.get('MGround')
        if _MGround is not None:
            output.write(f' MGround={_MGround}')

        _EventLog = fields.get('EventLog')
        if _EventLog is not None:
            output.write(f' EventLog={_EventLog}')

        _DebugTrace = fields.get('DebugTrace')
        if _DebugTrace is not None:
            output.write(f' DebugTrace={_DebugTrace}')

        _DistReverse = fields.get('DistReverse')
        if _DistReverse is not None:
            output.write(f' DistReverse={_DistReverse}')

        _Normal = fields.get('Normal')
        if _Normal is not None:
            output.write(f' Normal={_quoted(_Normal)}')

        _State = fields.get('State')
        if _State is not None:
            output.write(f' State={_quoted(_State)}')

        _DOC_TiltAngleLow = fields.get('DOC_TiltAngleLow')
        if _DOC_TiltAngleLow is not None:
            output.write(f' DOC_TiltAngleLow={_DOC_TiltAngleLow}')

        _DOC_TiltAngleHigh = fields.get('DOC_TiltAngleHigh')
        if _DOC_TiltAngleHigh is not None:
            output.write(f' DOC_TiltAngleHigh={_DOC_TiltAngleHigh}')

        _DOC_TripSettingLow = fields.get('DOC_TripSettingLow')
        if _DOC_TripSettingLow is not None:
            output.write(f' DOC_TripSettingLow={_DOC_TripSettingLow}')

        _DOC_TripSettingHigh = fields.get('DOC_TripSettingHigh')
        if _DOC_TripSettingHigh is not None:
            output.write(f' DOC_TripSettingHigh={_DOC_TripSettingHigh}')

        _DOC_TripSettingMag = fields.get('DOC_TripSettingMag')
        if _DOC_TripSettingMag is not None:
            output.write(f' DOC_TripSettingMag={_DOC_TripSettingMag}')

        _DOC_DelayInner = fields.get('DOC_DelayInner')
        if _DOC_DelayInner is not None:
            output.write(f' DOC_DelayInner={_DOC_DelayInner}')

        _DOC_PhaseCurveInner = fields.get('DOC_PhaseCurveInner')
        if _DOC_PhaseCurveInner is not None:
            output.write(f' DOC_PhaseCurveInner={_quoted(_DOC_PhaseCurveInner)}')

        _DOC_PhaseTripInner = fields.get('DOC_PhaseTripInner')
        if _DOC_PhaseTripInner is not None:
            output.write(f' DOC_PhaseTripInner={_DOC_PhaseTripInner}')

        _DOC_TDPhaseInner = fields.get('DOC_TDPhaseInner')
        if _DOC_TDPhaseInner is not None:
            output.write(f' DOC_TDPhaseInner={_DOC_TDPhaseInner}')

        _DOC_P1Blocking = fields.get('DOC_P1Blocking')
        if _DOC_P1Blocking is not None:
            output.write(f' DOC_P1Blocking={_DOC_P1Blocking}')

        _BaseFreq = fields.get('BaseFreq')
        if _BaseFreq is not None:
            output.write(f' BaseFreq={_BaseFreq}')

        _Enabled = fields.get('Enabled')
        if _Enabled is not None:
            output.write(f' Enabled={_Enabled}')

        output.write('\n')


Relay_ = Relay


class RelayList(RootModel[List[Relay]]):
    root: List[Relay]





class RelayContainer(RootModel[Union[RelayList, JSONFilePath, JSONLinesFilePath]]):
    root: Union[RelayList, JSONFilePath, JSONLinesFilePath] = Field(..., title="RelayContainer")


    @model_validator(mode="before")
    def _val_dss_model(self) -> "RelayContainer":
        try:
            _fields_set = set(self.keys())
        except:
            if isinstance(self, (list, tuple)):
                return self

            raise

        # Validate oneOf (spec. sets) based on the `required` lists
        _required_RelayList = "root" in _fields_set and isinstance(self.root, RelayList)
        _required_JSONFilePath = _fields_set.issuperset({'JSONFile'})
        _required_JSONLinesFilePath = _fields_set.issuperset({'JSONLinesFile'})
        num_specs = _required_RelayList + _required_JSONFilePath + _required_JSONLinesFilePath
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



class Recloser(BaseModel):
    Name: Annotated[str, Field(max_length=255, min_length=1)] = Field(..., title="Name")
    Like: Optional[str] = Field(None, title="Like")
    MonitoredObj: str = Field(..., title="MonitoredObj")
    MonitoredTerm: Optional[int] = Field(None, title="MonitoredTerm")
    SwitchedObj: Optional[str] = Field(None, title="SwitchedObj")
    SwitchedTerm: Optional[int] = Field(None, title="SwitchedTerm")
    NumFast: Optional[int] = Field(None, title="NumFast")
    PhaseFast: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="PhaseFast")
    PhaseDelayed: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="PhaseDelayed")
    GroundFast: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="GroundFast")
    GroundDelayed: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="GroundDelayed")
    PhaseTrip: Optional[float] = Field(None, title="PhaseTrip")
    GroundTrip: Optional[float] = Field(None, title="GroundTrip")
    PhaseInst: Optional[float] = Field(None, title="PhaseInst")
    GroundInst: Optional[float] = Field(None, title="GroundInst")
    Reset: Optional[float] = Field(None, title="Reset")
    Shots: Optional[PositiveInt] = Field(None, title="Shots")
    RecloseIntervals: Optional[List[float]] = Field(None, title="RecloseIntervals")
    Delay: Optional[float] = Field(None, title="Delay")
    TDPhFast: Optional[float] = Field(None, title="TDPhFast")
    TDGrFast: Optional[float] = Field(None, title="TDGrFast")
    TDPhDelayed: Optional[float] = Field(None, title="TDPhDelayed")
    TDGrDelayed: Optional[float] = Field(None, title="TDGrDelayed")
    Normal: Optional[RecloserState] = Field(None, title="Normal")
    State: Optional[RecloserState] = Field(None, title="State")
    BaseFreq: Optional[PositiveFloat] = Field(None, title="BaseFreq")
    Enabled: Optional[bool] = Field(None, title="Enabled")

    def dumps_dss(self, edit=False) -> str:
        """
        Dump the DSS object dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            self.dump_dss(output, edit)
            return output.getvalue()

    def dump_dss(self, output: TextIO, edit=False):
        """
        Dump the DSS object to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        Recloser.dict_dump_dss(self.model_dump(exclude_unset=True), output, edit=edit)

    @staticmethod
    def dict_dumps_dss(fields: Dict, edit=False) -> str:
        """
        Dump the DSS object from a dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            Recloser.dict_dump_dss(fields, output, edit)
            return output.getvalue()

    @staticmethod
    def dict_dump_dss(fields: Dict, output: TextIO, edit=False):
        """
        Dump the DSS object from a dict to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        if edit:
            output.write(f'''edit Recloser.{fields['Name']}''')
        else:
            output.write(f'''new Recloser.{fields['Name']}''')

        # NOTE: "Shots" is redundant, left for clarity

        _Like = fields.get('Like')
        if _Like is not None:
            output.write(f' Like={_quoted(_Like)}')

        _MonitoredObj = fields.get('MonitoredObj')
        output.write(f' MonitoredObj={_quoted(_MonitoredObj)}')
        _MonitoredTerm = fields.get('MonitoredTerm')
        if _MonitoredTerm is not None:
            output.write(f' MonitoredTerm={_MonitoredTerm}')

        _SwitchedObj = fields.get('SwitchedObj')
        if _SwitchedObj is not None:
            output.write(f' SwitchedObj={_quoted(_SwitchedObj)}')

        _SwitchedTerm = fields.get('SwitchedTerm')
        if _SwitchedTerm is not None:
            output.write(f' SwitchedTerm={_SwitchedTerm}')

        _NumFast = fields.get('NumFast')
        if _NumFast is not None:
            output.write(f' NumFast={_NumFast}')

        _PhaseFast = fields.get('PhaseFast')
        if _PhaseFast is not None:
            output.write(f' PhaseFast={_quoted(_PhaseFast)}')

        _PhaseDelayed = fields.get('PhaseDelayed')
        if _PhaseDelayed is not None:
            output.write(f' PhaseDelayed={_quoted(_PhaseDelayed)}')

        _GroundFast = fields.get('GroundFast')
        if _GroundFast is not None:
            output.write(f' GroundFast={_quoted(_GroundFast)}')

        _GroundDelayed = fields.get('GroundDelayed')
        if _GroundDelayed is not None:
            output.write(f' GroundDelayed={_quoted(_GroundDelayed)}')

        _PhaseTrip = fields.get('PhaseTrip')
        if _PhaseTrip is not None:
            output.write(f' PhaseTrip={_PhaseTrip}')

        _GroundTrip = fields.get('GroundTrip')
        if _GroundTrip is not None:
            output.write(f' GroundTrip={_GroundTrip}')

        _PhaseInst = fields.get('PhaseInst')
        if _PhaseInst is not None:
            output.write(f' PhaseInst={_PhaseInst}')

        _GroundInst = fields.get('GroundInst')
        if _GroundInst is not None:
            output.write(f' GroundInst={_GroundInst}')

        _Reset = fields.get('Reset')
        if _Reset is not None:
            output.write(f' Reset={_Reset}')

        _Shots = fields.get('Shots')
        if _Shots is not None:
            _length_Shots = _Shots - 1
            output.write(f' Shots={_Shots}')

        _RecloseIntervals = fields.get('RecloseIntervals')
        if _RecloseIntervals is not None:
            if len(_RecloseIntervals) != _length_Shots:
                raise ValueError(f'Array length ({len(_RecloseIntervals)}) for "RecloseIntervals" does not match expected length ({_length_Shots})')

            output.write(f' RecloseIntervals={_as_list(_RecloseIntervals)}')

        _Delay = fields.get('Delay')
        if _Delay is not None:
            output.write(f' Delay={_Delay}')

        _TDPhFast = fields.get('TDPhFast')
        if _TDPhFast is not None:
            output.write(f' TDPhFast={_TDPhFast}')

        _TDGrFast = fields.get('TDGrFast')
        if _TDGrFast is not None:
            output.write(f' TDGrFast={_TDGrFast}')

        _TDPhDelayed = fields.get('TDPhDelayed')
        if _TDPhDelayed is not None:
            output.write(f' TDPhDelayed={_TDPhDelayed}')

        _TDGrDelayed = fields.get('TDGrDelayed')
        if _TDGrDelayed is not None:
            output.write(f' TDGrDelayed={_TDGrDelayed}')

        _Normal = fields.get('Normal')
        if _Normal is not None:
            output.write(f' Normal={_quoted(_Normal)}')

        _State = fields.get('State')
        if _State is not None:
            output.write(f' State={_quoted(_State)}')

        _BaseFreq = fields.get('BaseFreq')
        if _BaseFreq is not None:
            output.write(f' BaseFreq={_BaseFreq}')

        _Enabled = fields.get('Enabled')
        if _Enabled is not None:
            output.write(f' Enabled={_Enabled}')

        output.write('\n')


Recloser_ = Recloser


class RecloserList(RootModel[List[Recloser]]):
    root: List[Recloser]





class RecloserContainer(RootModel[Union[RecloserList, JSONFilePath, JSONLinesFilePath]]):
    root: Union[RecloserList, JSONFilePath, JSONLinesFilePath] = Field(..., title="RecloserContainer")


    @model_validator(mode="before")
    def _val_dss_model(self) -> "RecloserContainer":
        try:
            _fields_set = set(self.keys())
        except:
            if isinstance(self, (list, tuple)):
                return self

            raise

        # Validate oneOf (spec. sets) based on the `required` lists
        _required_RecloserList = "root" in _fields_set and isinstance(self.root, RecloserList)
        _required_JSONFilePath = _fields_set.issuperset({'JSONFile'})
        _required_JSONLinesFilePath = _fields_set.issuperset({'JSONLinesFile'})
        num_specs = _required_RecloserList + _required_JSONFilePath + _required_JSONLinesFilePath
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



class Fuse(BaseModel):
    Name: Annotated[str, Field(max_length=255, min_length=1)] = Field(..., title="Name")
    Like: Optional[str] = Field(None, title="Like")
    MonitoredObj: str = Field(..., title="MonitoredObj")
    MonitoredTerm: Optional[int] = Field(None, title="MonitoredTerm")
    SwitchedObj: Optional[str] = Field(None, title="SwitchedObj")
    SwitchedTerm: Optional[int] = Field(None, title="SwitchedTerm")
    FuseCurve: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="FuseCurve")
    RatedCurrent: Optional[float] = Field(None, title="RatedCurrent")
    Delay: Optional[float] = Field(None, title="Delay")
    Normal: Optional[List[FuseState]] = Field(None, title="Normal")
    State: Optional[List[FuseState]] = Field(None, title="State")
    BaseFreq: Optional[PositiveFloat] = Field(None, title="BaseFreq")
    Enabled: Optional[bool] = Field(None, title="Enabled")
    Action: Optional[FuseAction] = Field(None, title="Action")

    def dumps_dss(self, edit=False) -> str:
        """
        Dump the DSS object dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            self.dump_dss(output, edit)
            return output.getvalue()

    def dump_dss(self, output: TextIO, edit=False):
        """
        Dump the DSS object to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        Fuse.dict_dump_dss(self.model_dump(exclude_unset=True), output, edit=edit)

    @staticmethod
    def dict_dumps_dss(fields: Dict, edit=False) -> str:
        """
        Dump the DSS object from a dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            Fuse.dict_dump_dss(fields, output, edit)
            return output.getvalue()

    @staticmethod
    def dict_dump_dss(fields: Dict, output: TextIO, edit=False):
        """
        Dump the DSS object from a dict to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        if edit:
            output.write(f'''edit Fuse.{fields['Name']}''')
        else:
            output.write(f'''new Fuse.{fields['Name']}''')

        _Like = fields.get('Like')
        if _Like is not None:
            output.write(f' Like={_quoted(_Like)}')

        _MonitoredObj = fields.get('MonitoredObj')
        output.write(f' MonitoredObj={_quoted(_MonitoredObj)}')
        _MonitoredTerm = fields.get('MonitoredTerm')
        if _MonitoredTerm is not None:
            output.write(f' MonitoredTerm={_MonitoredTerm}')

        _SwitchedObj = fields.get('SwitchedObj')
        if _SwitchedObj is not None:
            output.write(f' SwitchedObj={_quoted(_SwitchedObj)}')

        _SwitchedTerm = fields.get('SwitchedTerm')
        if _SwitchedTerm is not None:
            output.write(f' SwitchedTerm={_SwitchedTerm}')

        _FuseCurve = fields.get('FuseCurve')
        if _FuseCurve is not None:
            output.write(f' FuseCurve={_quoted(_FuseCurve)}')

        _RatedCurrent = fields.get('RatedCurrent')
        if _RatedCurrent is not None:
            output.write(f' RatedCurrent={_RatedCurrent}')

        _Delay = fields.get('Delay')
        if _Delay is not None:
            output.write(f' Delay={_Delay}')

        _Normal = fields.get('Normal')
        if _Normal is not None:
            output.write(f' Normal=({_quoted_list(_Normal)})')

        _State = fields.get('State')
        if _State is not None:
            output.write(f' State=({_quoted_list(_State)})')

        _BaseFreq = fields.get('BaseFreq')
        if _BaseFreq is not None:
            output.write(f' BaseFreq={_BaseFreq}')

        _Enabled = fields.get('Enabled')
        if _Enabled is not None:
            output.write(f' Enabled={_Enabled}')

        _Action = fields.get('Action')
        if _Action is not None:
            output.write(f' Action={_quoted(_Action)}')

        output.write('\n')


Fuse_ = Fuse


class FuseList(RootModel[List[Fuse]]):
    root: List[Fuse]





class FuseContainer(RootModel[Union[FuseList, JSONFilePath, JSONLinesFilePath]]):
    root: Union[FuseList, JSONFilePath, JSONLinesFilePath] = Field(..., title="FuseContainer")


    @model_validator(mode="before")
    def _val_dss_model(self) -> "FuseContainer":
        try:
            _fields_set = set(self.keys())
        except:
            if isinstance(self, (list, tuple)):
                return self

            raise

        # Validate oneOf (spec. sets) based on the `required` lists
        _required_FuseList = "root" in _fields_set and isinstance(self.root, FuseList)
        _required_JSONFilePath = _fields_set.issuperset({'JSONFile'})
        _required_JSONLinesFilePath = _fields_set.issuperset({'JSONLinesFile'})
        num_specs = _required_FuseList + _required_JSONFilePath + _required_JSONLinesFilePath
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



class SwtControl(BaseModel):
    Name: Annotated[str, Field(max_length=255, min_length=1)] = Field(..., title="Name")
    Like: Optional[str] = Field(None, title="Like")
    SwitchedObj: Optional[str] = Field(None, title="SwitchedObj")
    SwitchedTerm: Optional[int] = Field(None, title="SwitchedTerm")
    Lock: Optional[bool] = Field(None, title="Lock")
    Delay: Optional[float] = Field(None, title="Delay")
    Normal: Optional[SwtControlState] = Field(None, title="Normal")
    State: Optional[SwtControlState] = Field(None, title="State")
    BaseFreq: Optional[PositiveFloat] = Field(None, title="BaseFreq")
    Enabled: Optional[bool] = Field(None, title="Enabled")
    Reset: Optional[bool] = Field(None, title="Reset")

    def dumps_dss(self, edit=False) -> str:
        """
        Dump the DSS object dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            self.dump_dss(output, edit)
            return output.getvalue()

    def dump_dss(self, output: TextIO, edit=False):
        """
        Dump the DSS object to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        SwtControl.dict_dump_dss(self.model_dump(exclude_unset=True), output, edit=edit)

    @staticmethod
    def dict_dumps_dss(fields: Dict, edit=False) -> str:
        """
        Dump the DSS object from a dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            SwtControl.dict_dump_dss(fields, output, edit)
            return output.getvalue()

    @staticmethod
    def dict_dump_dss(fields: Dict, output: TextIO, edit=False):
        """
        Dump the DSS object from a dict to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        if edit:
            output.write(f'''edit SwtControl.{fields['Name']}''')
        else:
            output.write(f'''new SwtControl.{fields['Name']}''')

        _Like = fields.get('Like')
        if _Like is not None:
            output.write(f' Like={_quoted(_Like)}')

        _SwitchedObj = fields.get('SwitchedObj')
        if _SwitchedObj is not None:
            output.write(f' SwitchedObj={_quoted(_SwitchedObj)}')

        _SwitchedTerm = fields.get('SwitchedTerm')
        if _SwitchedTerm is not None:
            output.write(f' SwitchedTerm={_SwitchedTerm}')

        _Lock = fields.get('Lock')
        if _Lock is not None:
            output.write(f' Lock={_Lock}')

        _Delay = fields.get('Delay')
        if _Delay is not None:
            output.write(f' Delay={_Delay}')

        _Normal = fields.get('Normal')
        if _Normal is not None:
            output.write(f' Normal={_quoted(_Normal)}')

        _State = fields.get('State')
        if _State is not None:
            output.write(f' State={_quoted(_State)}')

        _BaseFreq = fields.get('BaseFreq')
        if _BaseFreq is not None:
            output.write(f' BaseFreq={_BaseFreq}')

        _Enabled = fields.get('Enabled')
        if _Enabled is not None:
            output.write(f' Enabled={_Enabled}')

        _Reset = fields.get('Reset')
        if _Reset is not None:
            output.write(f' Reset={_Reset}')

        output.write('\n')


SwtControl_ = SwtControl


class SwtControlList(RootModel[List[SwtControl]]):
    root: List[SwtControl]





class SwtControlContainer(RootModel[Union[SwtControlList, JSONFilePath, JSONLinesFilePath]]):
    root: Union[SwtControlList, JSONFilePath, JSONLinesFilePath] = Field(..., title="SwtControlContainer")


    @model_validator(mode="before")
    def _val_dss_model(self) -> "SwtControlContainer":
        try:
            _fields_set = set(self.keys())
        except:
            if isinstance(self, (list, tuple)):
                return self

            raise

        # Validate oneOf (spec. sets) based on the `required` lists
        _required_SwtControlList = "root" in _fields_set and isinstance(self.root, SwtControlList)
        _required_JSONFilePath = _fields_set.issuperset({'JSONFile'})
        _required_JSONLinesFilePath = _fields_set.issuperset({'JSONLinesFile'})
        num_specs = _required_SwtControlList + _required_JSONFilePath + _required_JSONLinesFilePath
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



class PVSystem_Common(BaseModel):
    Name: Annotated[str, Field(max_length=255, min_length=1)] = Field(..., title="Name")
    Like: Optional[str] = Field(None, title="Like")
    Phases: Optional[PositiveInt] = Field(None, title="Phases")
    Bus1: BusConnection = Field(..., title="Bus1")
    kV: Annotated[float, Field(ge=0)] = Field(..., title="kV")
    Irradiance: Optional[float] = Field(None, title="Irradiance")
    Pmpp: Optional[float] = Field(None, title="Pmpp")
    pctPmpp: Optional[float] = Field(None, title="%Pmpp", validation_alias=AliasChoices("pctPmpp", "%Pmpp"))
    Temperature: Optional[float] = Field(None, title="Temperature")
    Conn: Optional[Connection] = Field(None, title="Conn")
    kVA: Optional[float] = Field(None, title="kVA")
    pctCutIn: Optional[float] = Field(None, title="%CutIn", validation_alias=AliasChoices("pctCutIn", "%CutIn"))
    pctCutOut: Optional[float] = Field(None, title="%CutOut", validation_alias=AliasChoices("pctCutOut", "%CutOut"))
    EffCurve: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="EffCurve")
    PTCurve: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="P-TCurve", validation_alias=AliasChoices("PTCurve", "P-TCurve"))
    pctR: Optional[float] = Field(None, title="%R", validation_alias=AliasChoices("pctR", "%R"))
    pctX: Optional[float] = Field(None, title="%X", validation_alias=AliasChoices("pctX", "%X"))
    Model: Optional[PVSystemModel] = Field(None, title="Model")
    VMinpu: Optional[float] = Field(None, title="VMinpu")
    VMaxpu: Optional[float] = Field(None, title="VMaxpu")
    Balanced: Optional[bool] = Field(None, title="Balanced")
    LimitCurrent: Optional[bool] = Field(None, title="LimitCurrent")
    Yearly: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="Yearly")
    Daily: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="Daily")
    Duty: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="Duty")
    TYearly: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="TYearly")
    TDaily: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="TDaily")
    TDuty: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="TDuty")
    Class: Optional[int] = Field(None, title="Class")
    UserModel: Optional[FilePath] = Field(None, title="UserModel")
    UserData: Optional[str] = Field(None, title="UserData")
    DebugTrace: Optional[bool] = Field(None, title="DebugTrace")
    VarFollowInverter: Optional[bool] = Field(None, title="VarFollowInverter")
    DutyStart: Optional[Annotated[float, Field(ge=0)]] = Field(None, title="DutyStart")
    WattPriority: Optional[bool] = Field(None, title="WattPriority")
    PFPriority: Optional[bool] = Field(None, title="PFPriority")
    pctPMinNoVars: Optional[float] = Field(None, title="%PMinNoVars", validation_alias=AliasChoices("pctPMinNoVars", "%PMinNoVars"))
    pctPMinkvarMax: Optional[float] = Field(None, title="%PMinkvarMax", validation_alias=AliasChoices("pctPMinkvarMax", "%PMinkvarMax"))
    kvarMax: Optional[float] = Field(None, title="kvarMax")
    kvarMaxAbs: Optional[float] = Field(None, title="kvarMaxAbs")
    kVDC: Optional[float] = Field(None, title="kVDC")
    Kp: Optional[float] = Field(None, title="Kp")
    PITol: Optional[float] = Field(None, title="PITol")
    SafeVoltage: Optional[float] = Field(None, title="SafeVoltage")
    DynamicEq: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="DynamicEq")
    DynOut: Optional[StringArrayOrFilePath] = Field(None, title="DynOut")
    ControlMode: Optional[InverterControlMode] = Field(None, title="ControlMode")
    AmpLimit: Optional[float] = Field(None, title="AmpLimit")
    AmpLimitGain: Optional[float] = Field(None, title="AmpLimitGain")
    Spectrum: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="Spectrum")
    BaseFreq: Optional[PositiveFloat] = Field(None, title="BaseFreq")
    Enabled: Optional[bool] = Field(None, title="Enabled")
    DynInit: Optional[DynInitType] = Field(None, title="DynInit")

class PVSystem_PF(PVSystem_Common):
    PF: Annotated[float, Field(le=1, ge=-1)] = Field(..., title="PF")

class PVSystem_kvar(PVSystem_Common):
    kvar: float = Field(..., title="kvar")


class PVSystem(RootModel[Union[PVSystem_PF, PVSystem_kvar]]):
    root: Union[PVSystem_PF, PVSystem_kvar] = Field(..., title="PVSystem")

    def dumps_dss(self, edit=False) -> str:
        """
        Dump the DSS object dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            self.dump_dss(output, edit)
            return output.getvalue()

    def dump_dss(self, output: TextIO, edit=False):
        """
        Dump the DSS object to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        PVSystem.dict_dump_dss(self.model_dump(exclude_unset=True), output, edit=edit)

    @staticmethod
    def dict_dumps_dss(fields: Dict, edit=False) -> str:
        """
        Dump the DSS object from a dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            PVSystem.dict_dump_dss(fields, output, edit)
            return output.getvalue()

    @staticmethod
    def dict_dump_dss(fields: Dict, output: TextIO, edit=False):
        """
        Dump the DSS object from a dict to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        if edit:
            output.write(f'''edit PVSystem.{fields['Name']}''')
        else:
            output.write(f'''new PVSystem.{fields['Name']}''')

        _Like = fields.get('Like')
        if _Like is not None:
            output.write(f' Like={_quoted(_Like)}')

        _Phases = fields.get('Phases')
        if _Phases is not None:
            output.write(f' Phases={_Phases}')

        _Bus1 = fields.get('Bus1')
        output.write(f' Bus1={_quoted(_Bus1)}')
        _kV = fields.get('kV')
        output.write(f' kV={_kV}')
        _Irradiance = fields.get('Irradiance')
        if _Irradiance is not None:
            output.write(f' Irradiance={_Irradiance}')

        _Pmpp = fields.get('Pmpp')
        if _Pmpp is not None:
            output.write(f' Pmpp={_Pmpp}')

        _pctPmpp = fields.get('pctPmpp')
        if _pctPmpp is not None:
            output.write(f' %Pmpp={_pctPmpp}')

        _Temperature = fields.get('Temperature')
        if _Temperature is not None:
            output.write(f' Temperature={_Temperature}')

        _PF = fields.get('PF')
        if _PF is not None:
            output.write(f' PF={_PF}')

        _Conn = fields.get('Conn')
        if _Conn is not None:
            output.write(f' Conn={_quoted(_Conn)}')

        _kvar = fields.get('kvar')
        if _kvar is not None:
            output.write(f' kvar={_kvar}')

        _kVA = fields.get('kVA')
        if _kVA is not None:
            output.write(f' kVA={_kVA}')

        _pctCutIn = fields.get('pctCutIn')
        if _pctCutIn is not None:
            output.write(f' %CutIn={_pctCutIn}')

        _pctCutOut = fields.get('pctCutOut')
        if _pctCutOut is not None:
            output.write(f' %CutOut={_pctCutOut}')

        _EffCurve = fields.get('EffCurve')
        if _EffCurve is not None:
            output.write(f' EffCurve={_quoted(_EffCurve)}')

        _PTCurve = fields.get('PTCurve')
        if _PTCurve is not None:
            output.write(f' P-TCurve={_quoted(_PTCurve)}')

        _pctR = fields.get('pctR')
        if _pctR is not None:
            output.write(f' %R={_pctR}')

        _pctX = fields.get('pctX')
        if _pctX is not None:
            output.write(f' %X={_pctX}')

        _Model = fields.get('Model')
        if _Model is not None:
            output.write(f' Model={_Model}')

        _VMinpu = fields.get('VMinpu')
        if _VMinpu is not None:
            output.write(f' VMinpu={_VMinpu}')

        _VMaxpu = fields.get('VMaxpu')
        if _VMaxpu is not None:
            output.write(f' VMaxpu={_VMaxpu}')

        _Balanced = fields.get('Balanced')
        if _Balanced is not None:
            output.write(f' Balanced={_Balanced}')

        _LimitCurrent = fields.get('LimitCurrent')
        if _LimitCurrent is not None:
            output.write(f' LimitCurrent={_LimitCurrent}')

        _Yearly = fields.get('Yearly')
        if _Yearly is not None:
            output.write(f' Yearly={_quoted(_Yearly)}')

        _Daily = fields.get('Daily')
        if _Daily is not None:
            output.write(f' Daily={_quoted(_Daily)}')

        _Duty = fields.get('Duty')
        if _Duty is not None:
            output.write(f' Duty={_quoted(_Duty)}')

        _TYearly = fields.get('TYearly')
        if _TYearly is not None:
            output.write(f' TYearly={_quoted(_TYearly)}')

        _TDaily = fields.get('TDaily')
        if _TDaily is not None:
            output.write(f' TDaily={_quoted(_TDaily)}')

        _TDuty = fields.get('TDuty')
        if _TDuty is not None:
            output.write(f' TDuty={_quoted(_TDuty)}')

        _Class = fields.get('Class')
        if _Class is not None:
            output.write(f' Class={_Class}')

        _UserModel = fields.get('UserModel')
        if _UserModel is not None:
            output.write(f' UserModel={_quoted(_UserModel)}')

        _UserData = fields.get('UserData')
        if _UserData is not None:
            output.write(f' UserData={_quoted(_UserData)}')

        _DebugTrace = fields.get('DebugTrace')
        if _DebugTrace is not None:
            output.write(f' DebugTrace={_DebugTrace}')

        _VarFollowInverter = fields.get('VarFollowInverter')
        if _VarFollowInverter is not None:
            output.write(f' VarFollowInverter={_VarFollowInverter}')

        _DutyStart = fields.get('DutyStart')
        if _DutyStart is not None:
            output.write(f' DutyStart={_DutyStart}')

        _WattPriority = fields.get('WattPriority')
        if _WattPriority is not None:
            output.write(f' WattPriority={_WattPriority}')

        _PFPriority = fields.get('PFPriority')
        if _PFPriority is not None:
            output.write(f' PFPriority={_PFPriority}')

        _pctPMinNoVars = fields.get('pctPMinNoVars')
        if _pctPMinNoVars is not None:
            output.write(f' %PMinNoVars={_pctPMinNoVars}')

        _pctPMinkvarMax = fields.get('pctPMinkvarMax')
        if _pctPMinkvarMax is not None:
            output.write(f' %PMinkvarMax={_pctPMinkvarMax}')

        _kvarMax = fields.get('kvarMax')
        if _kvarMax is not None:
            output.write(f' kvarMax={_kvarMax}')

        _kvarMaxAbs = fields.get('kvarMaxAbs')
        if _kvarMaxAbs is not None:
            output.write(f' kvarMaxAbs={_kvarMaxAbs}')

        _kVDC = fields.get('kVDC')
        if _kVDC is not None:
            output.write(f' kVDC={_kVDC}')

        _Kp = fields.get('Kp')
        if _Kp is not None:
            output.write(f' Kp={_Kp}')

        _PITol = fields.get('PITol')
        if _PITol is not None:
            output.write(f' PITol={_PITol}')

        _SafeVoltage = fields.get('SafeVoltage')
        if _SafeVoltage is not None:
            output.write(f' SafeVoltage={_SafeVoltage}')

        _DynamicEq = fields.get('DynamicEq')
        if _DynamicEq is not None:
            output.write(f' DynamicEq={_quoted(_DynamicEq)}')

        _DynOut = fields.get('DynOut')
        if _DynOut is not None:
            output.write(f' DynOut=({_filepath_stringlist(_DynOut)})')

        _ControlMode = fields.get('ControlMode')
        if _ControlMode is not None:
            output.write(f' ControlMode={_quoted(_ControlMode)}')

        _AmpLimit = fields.get('AmpLimit')
        if _AmpLimit is not None:
            output.write(f' AmpLimit={_AmpLimit}')

        _AmpLimitGain = fields.get('AmpLimitGain')
        if _AmpLimitGain is not None:
            output.write(f' AmpLimitGain={_AmpLimitGain}')

        _Spectrum = fields.get('Spectrum')
        if _Spectrum is not None:
            output.write(f' Spectrum={_quoted(_Spectrum)}')

        _BaseFreq = fields.get('BaseFreq')
        if _BaseFreq is not None:
            output.write(f' BaseFreq={_BaseFreq}')

        _Enabled = fields.get('Enabled')
        if _Enabled is not None:
            output.write(f' Enabled={_Enabled}')

        _DynInit = fields.get('DynInit')
        if _DynInit is not None:
            DynInitType.dict_dump_dss(_DynInit, output)

        output.write('\n')

    @model_validator(mode="before")
    def _val_dss_model(self) -> "PVSystem":
        _fields_set = set(self.keys())
        # Validate oneOf (spec. sets) based on the `required` lists
        _required_PF = _fields_set.issuperset({'PF'})
        _required_kvar = _fields_set.issuperset({'kvar'})
        num_specs = _required_PF + _required_kvar
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



PVSystem_ = PVSystem


class PVSystemList(RootModel[List[PVSystem]]):
    root: List[PVSystem]



PVSystemList_ = PVSystemList




class PVSystemContainer(RootModel[Union[PVSystemList, JSONFilePath, JSONLinesFilePath]]):
    root: Union[PVSystemList, JSONFilePath, JSONLinesFilePath] = Field(..., title="PVSystemContainer")


    @model_validator(mode="before")
    def _val_dss_model(self) -> "PVSystemContainer":
        try:
            _fields_set = set(self.keys())
        except:
            if isinstance(self, (list, tuple)):
                return self

            raise

        # Validate oneOf (spec. sets) based on the `required` lists
        _required_PVSystemList = "root" in _fields_set and isinstance(self.root, PVSystemList)
        _required_JSONFilePath = _fields_set.issuperset({'JSONFile'})
        _required_JSONLinesFilePath = _fields_set.issuperset({'JSONLinesFile'})
        num_specs = _required_PVSystemList + _required_JSONFilePath + _required_JSONLinesFilePath
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



class UPFC(BaseModel):
    Name: Annotated[str, Field(max_length=255, min_length=1)] = Field(..., title="Name")
    Like: Optional[str] = Field(None, title="Like")
    Bus1: BusConnection = Field(..., title="Bus1")
    Bus2: BusConnection = Field(..., title="Bus2")
    RefkV: Optional[float] = Field(None, title="RefkV")
    PF: Optional[Annotated[float, Field(le=1, ge=-1)]] = Field(None, title="PF")
    Frequency: Optional[PositiveFloat] = Field(None, title="Frequency")
    Phases: Optional[PositiveInt] = Field(None, title="Phases")
    Xs: Optional[float] = Field(None, title="Xs")
    Tol1: Optional[float] = Field(None, title="Tol1")
    Mode: Optional[UPFCMode] = Field(None, title="Mode")
    VpqMax: Optional[float] = Field(None, title="VpqMax")
    LossCurve: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="LossCurve")
    VHLimit: Optional[float] = Field(None, title="VHLimit")
    VLLimit: Optional[float] = Field(None, title="VLLimit")
    CLimit: Optional[float] = Field(None, title="CLimit")
    refkV2: Optional[float] = Field(None, title="refkV2")
    kvarLimit: Optional[float] = Field(None, title="kvarLimit")
    Element: Optional[str] = Field(None, title="Element")
    Spectrum: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="Spectrum")
    BaseFreq: Optional[PositiveFloat] = Field(None, title="BaseFreq")
    Enabled: Optional[bool] = Field(None, title="Enabled")

    def dumps_dss(self, edit=False) -> str:
        """
        Dump the DSS object dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            self.dump_dss(output, edit)
            return output.getvalue()

    def dump_dss(self, output: TextIO, edit=False):
        """
        Dump the DSS object to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        UPFC.dict_dump_dss(self.model_dump(exclude_unset=True), output, edit=edit)

    @staticmethod
    def dict_dumps_dss(fields: Dict, edit=False) -> str:
        """
        Dump the DSS object from a dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            UPFC.dict_dump_dss(fields, output, edit)
            return output.getvalue()

    @staticmethod
    def dict_dump_dss(fields: Dict, output: TextIO, edit=False):
        """
        Dump the DSS object from a dict to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        if edit:
            output.write(f'''edit UPFC.{fields['Name']}''')
        else:
            output.write(f'''new UPFC.{fields['Name']}''')

        _Like = fields.get('Like')
        if _Like is not None:
            output.write(f' Like={_quoted(_Like)}')

        _Bus1 = fields.get('Bus1')
        output.write(f' Bus1={_quoted(_Bus1)}')
        _Bus2 = fields.get('Bus2')
        output.write(f' Bus2={_quoted(_Bus2)}')
        _RefkV = fields.get('RefkV')
        if _RefkV is not None:
            output.write(f' RefkV={_RefkV}')

        _PF = fields.get('PF')
        if _PF is not None:
            output.write(f' PF={_PF}')

        _Frequency = fields.get('Frequency')
        if _Frequency is not None:
            output.write(f' Frequency={_Frequency}')

        _Phases = fields.get('Phases')
        if _Phases is not None:
            output.write(f' Phases={_Phases}')

        _Xs = fields.get('Xs')
        if _Xs is not None:
            output.write(f' Xs={_Xs}')

        _Tol1 = fields.get('Tol1')
        if _Tol1 is not None:
            output.write(f' Tol1={_Tol1}')

        _Mode = fields.get('Mode')
        if _Mode is not None:
            output.write(f' Mode={_Mode}')

        _VpqMax = fields.get('VpqMax')
        if _VpqMax is not None:
            output.write(f' VpqMax={_VpqMax}')

        _LossCurve = fields.get('LossCurve')
        if _LossCurve is not None:
            output.write(f' LossCurve={_quoted(_LossCurve)}')

        _VHLimit = fields.get('VHLimit')
        if _VHLimit is not None:
            output.write(f' VHLimit={_VHLimit}')

        _VLLimit = fields.get('VLLimit')
        if _VLLimit is not None:
            output.write(f' VLLimit={_VLLimit}')

        _CLimit = fields.get('CLimit')
        if _CLimit is not None:
            output.write(f' CLimit={_CLimit}')

        _refkV2 = fields.get('refkV2')
        if _refkV2 is not None:
            output.write(f' refkV2={_refkV2}')

        _kvarLimit = fields.get('kvarLimit')
        if _kvarLimit is not None:
            output.write(f' kvarLimit={_kvarLimit}')

        _Element = fields.get('Element')
        if _Element is not None:
            output.write(f' Element={_quoted(_Element)}')

        _Spectrum = fields.get('Spectrum')
        if _Spectrum is not None:
            output.write(f' Spectrum={_quoted(_Spectrum)}')

        _BaseFreq = fields.get('BaseFreq')
        if _BaseFreq is not None:
            output.write(f' BaseFreq={_BaseFreq}')

        _Enabled = fields.get('Enabled')
        if _Enabled is not None:
            output.write(f' Enabled={_Enabled}')

        output.write('\n')


UPFC_ = UPFC


class UPFCList(RootModel[List[UPFC]]):
    root: List[UPFC]



UPFCList_ = UPFCList




class UPFCContainer(RootModel[Union[UPFCList, JSONFilePath, JSONLinesFilePath]]):
    root: Union[UPFCList, JSONFilePath, JSONLinesFilePath] = Field(..., title="UPFCContainer")


    @model_validator(mode="before")
    def _val_dss_model(self) -> "UPFCContainer":
        try:
            _fields_set = set(self.keys())
        except:
            if isinstance(self, (list, tuple)):
                return self

            raise

        # Validate oneOf (spec. sets) based on the `required` lists
        _required_UPFCList = "root" in _fields_set and isinstance(self.root, UPFCList)
        _required_JSONFilePath = _fields_set.issuperset({'JSONFile'})
        _required_JSONLinesFilePath = _fields_set.issuperset({'JSONLinesFile'})
        num_specs = _required_UPFCList + _required_JSONFilePath + _required_JSONLinesFilePath
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



class UPFCControl(BaseModel):
    Name: Annotated[str, Field(max_length=255, min_length=1)] = Field(..., title="Name")
    Like: Optional[str] = Field(None, title="Like")
    UPFCList: Optional[StringArrayOrFilePath] = Field(None, title="UPFCList")
    BaseFreq: Optional[PositiveFloat] = Field(None, title="BaseFreq")
    Enabled: Optional[bool] = Field(None, title="Enabled")

    def dumps_dss(self, edit=False) -> str:
        """
        Dump the DSS object dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            self.dump_dss(output, edit)
            return output.getvalue()

    def dump_dss(self, output: TextIO, edit=False):
        """
        Dump the DSS object to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        UPFCControl.dict_dump_dss(self.model_dump(exclude_unset=True), output, edit=edit)

    @staticmethod
    def dict_dumps_dss(fields: Dict, edit=False) -> str:
        """
        Dump the DSS object from a dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            UPFCControl.dict_dump_dss(fields, output, edit)
            return output.getvalue()

    @staticmethod
    def dict_dump_dss(fields: Dict, output: TextIO, edit=False):
        """
        Dump the DSS object from a dict to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        if edit:
            output.write(f'''edit UPFCControl.{fields['Name']}''')
        else:
            output.write(f'''new UPFCControl.{fields['Name']}''')

        _Like = fields.get('Like')
        if _Like is not None:
            output.write(f' Like={_quoted(_Like)}')

        _UPFCList = fields.get('UPFCList')
        if _UPFCList is not None:
            output.write(f' UPFCList=({_filepath_stringlist(_UPFCList)})')

        _BaseFreq = fields.get('BaseFreq')
        if _BaseFreq is not None:
            output.write(f' BaseFreq={_BaseFreq}')

        _Enabled = fields.get('Enabled')
        if _Enabled is not None:
            output.write(f' Enabled={_Enabled}')

        output.write('\n')


UPFCControl_ = UPFCControl


class UPFCControlList(RootModel[List[UPFCControl]]):
    root: List[UPFCControl]





class UPFCControlContainer(RootModel[Union[UPFCControlList, JSONFilePath, JSONLinesFilePath]]):
    root: Union[UPFCControlList, JSONFilePath, JSONLinesFilePath] = Field(..., title="UPFCControlContainer")


    @model_validator(mode="before")
    def _val_dss_model(self) -> "UPFCControlContainer":
        try:
            _fields_set = set(self.keys())
        except:
            if isinstance(self, (list, tuple)):
                return self

            raise

        # Validate oneOf (spec. sets) based on the `required` lists
        _required_UPFCControlList = "root" in _fields_set and isinstance(self.root, UPFCControlList)
        _required_JSONFilePath = _fields_set.issuperset({'JSONFile'})
        _required_JSONLinesFilePath = _fields_set.issuperset({'JSONLinesFile'})
        num_specs = _required_UPFCControlList + _required_JSONFilePath + _required_JSONLinesFilePath
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



class ESPVLControl(BaseModel):
    Name: Annotated[str, Field(max_length=255, min_length=1)] = Field(..., title="Name")
    Like: Optional[str] = Field(None, title="Like")
    Element: str = Field(..., title="Element")
    Terminal: Optional[int] = Field(None, title="Terminal")
    Type: Optional[ESPVLControlType] = Field(None, title="Type")
    kWBand: Optional[float] = Field(None, title="kWBand")
    kvarLimit: Optional[float] = Field(None, title="kvarLimit")
    LocalControlList: Optional[StringArrayOrFilePath] = Field(None, title="LocalControlList")
    LocalControlWeights: Optional[ArrayOrFilePath] = Field(None, title="LocalControlWeights")
    PVSystemList: Optional[StringArrayOrFilePath] = Field(None, title="PVSystemList")
    PVSystemWeights: Optional[ArrayOrFilePath] = Field(None, title="PVSystemWeights")
    StorageList: Optional[StringArrayOrFilePath] = Field(None, title="StorageList")
    StorageWeights: Optional[ArrayOrFilePath] = Field(None, title="StorageWeights")
    BaseFreq: Optional[PositiveFloat] = Field(None, title="BaseFreq")
    Enabled: Optional[bool] = Field(None, title="Enabled")

    def dumps_dss(self, edit=False) -> str:
        """
        Dump the DSS object dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            self.dump_dss(output, edit)
            return output.getvalue()

    def dump_dss(self, output: TextIO, edit=False):
        """
        Dump the DSS object to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        ESPVLControl.dict_dump_dss(self.model_dump(exclude_unset=True), output, edit=edit)

    @staticmethod
    def dict_dumps_dss(fields: Dict, edit=False) -> str:
        """
        Dump the DSS object from a dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            ESPVLControl.dict_dump_dss(fields, output, edit)
            return output.getvalue()

    @staticmethod
    def dict_dump_dss(fields: Dict, output: TextIO, edit=False):
        """
        Dump the DSS object from a dict to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        if edit:
            output.write(f'''edit ESPVLControl.{fields['Name']}''')
        else:
            output.write(f'''new ESPVLControl.{fields['Name']}''')

        # NOTE: "LocalControlList" is redundant, left for clarity

        # NOTE: "PVSystemList" is redundant, left for clarity

        # NOTE: "StorageList" is redundant, left for clarity

        _Like = fields.get('Like')
        if _Like is not None:
            output.write(f' Like={_quoted(_Like)}')

        _Element = fields.get('Element')
        output.write(f' Element={_quoted(_Element)}')
        _Terminal = fields.get('Terminal')
        if _Terminal is not None:
            output.write(f' Terminal={_Terminal}')

        _Type = fields.get('Type')
        if _Type is not None:
            output.write(f' Type={_quoted(_Type)}')

        _kWBand = fields.get('kWBand')
        if _kWBand is not None:
            output.write(f' kWBand={_kWBand}')

        _kvarLimit = fields.get('kvarLimit')
        if _kvarLimit is not None:
            output.write(f' kvarLimit={_kvarLimit}')

        _LocalControlList = fields.get('LocalControlList')
        if _LocalControlList is not None:
            _length_LocalControlList, _LocalControlList = _filepath_stringlist(_LocalControlList, length=True)
            output.write(f' LocalControlList=({_LocalControlList})')

        _LocalControlWeights = fields.get('LocalControlWeights')
        if _LocalControlWeights is not None:
            if isinstance(_LocalControlWeights, ARRAY_LIKE):
                if len(_LocalControlWeights) != _length_LocalControlList:
                    raise ValueError(f'Array length ({len(_LocalControlWeights)}) for "LocalControlWeights" does not match expected length ({_length_LocalControlList})')

                _LocalControlWeights = _as_list(_LocalControlWeights)
            else:
                _length_LocalControlWeights, _LocalControlWeights = _filepath_array(_LocalControlWeights)
                if _length_LocalControlList != _length_LocalControlWeights:
                    raise ValueError(f'Array length ({_length_LocalControlWeights}) for "LocalControlWeights" (from file) does not match expected length ({_length_LocalControlList})')

            output.write(f' LocalControlWeights={_LocalControlWeights}')

        _PVSystemList = fields.get('PVSystemList')
        if _PVSystemList is not None:
            _length_PVSystemList, _PVSystemList = _filepath_stringlist(_PVSystemList, length=True)
            output.write(f' PVSystemList=({_PVSystemList})')

        _PVSystemWeights = fields.get('PVSystemWeights')
        if _PVSystemWeights is not None:
            if isinstance(_PVSystemWeights, ARRAY_LIKE):
                if len(_PVSystemWeights) != _length_PVSystemList:
                    raise ValueError(f'Array length ({len(_PVSystemWeights)}) for "PVSystemWeights" does not match expected length ({_length_PVSystemList})')

                _PVSystemWeights = _as_list(_PVSystemWeights)
            else:
                _length_PVSystemWeights, _PVSystemWeights = _filepath_array(_PVSystemWeights)
                if _length_PVSystemList != _length_PVSystemWeights:
                    raise ValueError(f'Array length ({_length_PVSystemWeights}) for "PVSystemWeights" (from file) does not match expected length ({_length_PVSystemList})')

            output.write(f' PVSystemWeights={_PVSystemWeights}')

        _StorageList = fields.get('StorageList')
        if _StorageList is not None:
            _length_StorageList, _StorageList = _filepath_stringlist(_StorageList, length=True)
            output.write(f' StorageList=({_StorageList})')

        _StorageWeights = fields.get('StorageWeights')
        if _StorageWeights is not None:
            if isinstance(_StorageWeights, ARRAY_LIKE):
                if len(_StorageWeights) != _length_StorageList:
                    raise ValueError(f'Array length ({len(_StorageWeights)}) for "StorageWeights" does not match expected length ({_length_StorageList})')

                _StorageWeights = _as_list(_StorageWeights)
            else:
                _length_StorageWeights, _StorageWeights = _filepath_array(_StorageWeights)
                if _length_StorageList != _length_StorageWeights:
                    raise ValueError(f'Array length ({_length_StorageWeights}) for "StorageWeights" (from file) does not match expected length ({_length_StorageList})')

            output.write(f' StorageWeights={_StorageWeights}')

        _BaseFreq = fields.get('BaseFreq')
        if _BaseFreq is not None:
            output.write(f' BaseFreq={_BaseFreq}')

        _Enabled = fields.get('Enabled')
        if _Enabled is not None:
            output.write(f' Enabled={_Enabled}')

        output.write('\n')


ESPVLControl_ = ESPVLControl


class ESPVLControlList(RootModel[List[ESPVLControl]]):
    root: List[ESPVLControl]





class ESPVLControlContainer(RootModel[Union[ESPVLControlList, JSONFilePath, JSONLinesFilePath]]):
    root: Union[ESPVLControlList, JSONFilePath, JSONLinesFilePath] = Field(..., title="ESPVLControlContainer")


    @model_validator(mode="before")
    def _val_dss_model(self) -> "ESPVLControlContainer":
        try:
            _fields_set = set(self.keys())
        except:
            if isinstance(self, (list, tuple)):
                return self

            raise

        # Validate oneOf (spec. sets) based on the `required` lists
        _required_ESPVLControlList = "root" in _fields_set and isinstance(self.root, ESPVLControlList)
        _required_JSONFilePath = _fields_set.issuperset({'JSONFile'})
        _required_JSONLinesFilePath = _fields_set.issuperset({'JSONLinesFile'})
        num_specs = _required_ESPVLControlList + _required_JSONFilePath + _required_JSONLinesFilePath
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



class IndMach012(BaseModel):
    Name: Annotated[str, Field(max_length=255, min_length=1)] = Field(..., title="Name")
    Like: Optional[str] = Field(None, title="Like")
    Phases: Optional[PositiveInt] = Field(None, title="Phases")
    Bus1: BusConnection = Field(..., title="Bus1")
    kV: Annotated[float, Field(ge=0)] = Field(..., title="kV")
    kW: float = Field(..., title="kW")
    Conn: Optional[Connection] = Field(None, title="Conn")
    kVA: float = Field(..., title="kVA")
    H: Optional[float] = Field(None, title="H")
    D: Optional[float] = Field(None, title="D")
    puRs: Optional[float] = Field(None, title="puRs")
    puXs: Optional[float] = Field(None, title="puXs")
    puRr: Optional[float] = Field(None, title="puRr")
    puXr: Optional[float] = Field(None, title="puXr")
    puXm: Optional[float] = Field(None, title="puXm")
    Slip: Optional[float] = Field(None, title="Slip")
    MaxSlip: Optional[float] = Field(None, title="MaxSlip")
    SlipOption: Optional[IndMach012SlipOption] = Field(None, title="SlipOption")
    Yearly: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="Yearly")
    Daily: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="Daily")
    Duty: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="Duty")
    DebugTrace: Optional[bool] = Field(None, title="DebugTrace")
    Spectrum: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="Spectrum")
    BaseFreq: Optional[PositiveFloat] = Field(None, title="BaseFreq")
    Enabled: Optional[bool] = Field(None, title="Enabled")

    def dumps_dss(self, edit=False) -> str:
        """
        Dump the DSS object dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            self.dump_dss(output, edit)
            return output.getvalue()

    def dump_dss(self, output: TextIO, edit=False):
        """
        Dump the DSS object to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        IndMach012.dict_dump_dss(self.model_dump(exclude_unset=True), output, edit=edit)

    @staticmethod
    def dict_dumps_dss(fields: Dict, edit=False) -> str:
        """
        Dump the DSS object from a dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            IndMach012.dict_dump_dss(fields, output, edit)
            return output.getvalue()

    @staticmethod
    def dict_dump_dss(fields: Dict, output: TextIO, edit=False):
        """
        Dump the DSS object from a dict to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        if edit:
            output.write(f'''edit IndMach012.{fields['Name']}''')
        else:
            output.write(f'''new IndMach012.{fields['Name']}''')

        _Like = fields.get('Like')
        if _Like is not None:
            output.write(f' Like={_quoted(_Like)}')

        _Phases = fields.get('Phases')
        if _Phases is not None:
            output.write(f' Phases={_Phases}')

        _Bus1 = fields.get('Bus1')
        output.write(f' Bus1={_quoted(_Bus1)}')
        _kV = fields.get('kV')
        output.write(f' kV={_kV}')
        _kW = fields.get('kW')
        output.write(f' kW={_kW}')
        _Conn = fields.get('Conn')
        if _Conn is not None:
            output.write(f' Conn={_quoted(_Conn)}')

        _kVA = fields.get('kVA')
        output.write(f' kVA={_kVA}')
        _H = fields.get('H')
        if _H is not None:
            output.write(f' H={_H}')

        _D = fields.get('D')
        if _D is not None:
            output.write(f' D={_D}')

        _puRs = fields.get('puRs')
        if _puRs is not None:
            output.write(f' puRs={_puRs}')

        _puXs = fields.get('puXs')
        if _puXs is not None:
            output.write(f' puXs={_puXs}')

        _puRr = fields.get('puRr')
        if _puRr is not None:
            output.write(f' puRr={_puRr}')

        _puXr = fields.get('puXr')
        if _puXr is not None:
            output.write(f' puXr={_puXr}')

        _puXm = fields.get('puXm')
        if _puXm is not None:
            output.write(f' puXm={_puXm}')

        _Slip = fields.get('Slip')
        if _Slip is not None:
            output.write(f' Slip={_Slip}')

        _MaxSlip = fields.get('MaxSlip')
        if _MaxSlip is not None:
            output.write(f' MaxSlip={_MaxSlip}')

        _SlipOption = fields.get('SlipOption')
        if _SlipOption is not None:
            output.write(f' SlipOption={_quoted(_SlipOption)}')

        _Yearly = fields.get('Yearly')
        if _Yearly is not None:
            output.write(f' Yearly={_quoted(_Yearly)}')

        _Daily = fields.get('Daily')
        if _Daily is not None:
            output.write(f' Daily={_quoted(_Daily)}')

        _Duty = fields.get('Duty')
        if _Duty is not None:
            output.write(f' Duty={_quoted(_Duty)}')

        _DebugTrace = fields.get('DebugTrace')
        if _DebugTrace is not None:
            output.write(f' DebugTrace={_DebugTrace}')

        _Spectrum = fields.get('Spectrum')
        if _Spectrum is not None:
            output.write(f' Spectrum={_quoted(_Spectrum)}')

        _BaseFreq = fields.get('BaseFreq')
        if _BaseFreq is not None:
            output.write(f' BaseFreq={_BaseFreq}')

        _Enabled = fields.get('Enabled')
        if _Enabled is not None:
            output.write(f' Enabled={_Enabled}')

        output.write('\n')


IndMach012_ = IndMach012


class IndMach012List(RootModel[List[IndMach012]]):
    root: List[IndMach012]





class IndMach012Container(RootModel[Union[IndMach012List, JSONFilePath, JSONLinesFilePath]]):
    root: Union[IndMach012List, JSONFilePath, JSONLinesFilePath] = Field(..., title="IndMach012Container")


    @model_validator(mode="before")
    def _val_dss_model(self) -> "IndMach012Container":
        try:
            _fields_set = set(self.keys())
        except:
            if isinstance(self, (list, tuple)):
                return self

            raise

        # Validate oneOf (spec. sets) based on the `required` lists
        _required_IndMach012List = "root" in _fields_set and isinstance(self.root, IndMach012List)
        _required_JSONFilePath = _fields_set.issuperset({'JSONFile'})
        _required_JSONLinesFilePath = _fields_set.issuperset({'JSONLinesFile'})
        num_specs = _required_IndMach012List + _required_JSONFilePath + _required_JSONLinesFilePath
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



class GICsource_Common(BaseModel):
    Name: Annotated[str, Field(max_length=255, min_length=1)] = Field(..., title="Name")
    Like: Optional[str] = Field(None, title="Like")
    Frequency: Optional[PositiveFloat] = Field(None, title="Frequency")
    Phases: Optional[PositiveInt] = Field(None, title="Phases")
    Spectrum: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="Spectrum")
    BaseFreq: Optional[PositiveFloat] = Field(None, title="BaseFreq")
    Enabled: Optional[bool] = Field(None, title="Enabled")

class GICsource_VoltsAngle(GICsource_Common):
    Volts: float = Field(..., title="Volts")
    Angle: Optional[float] = Field(None, title="Angle")

class GICsource_ENEELat1Lon1Lat2Lon2(GICsource_Common):
    EN: float = Field(..., title="EN")
    EE: float = Field(..., title="EE")
    Lat1: float = Field(..., title="Lat1")
    Lon1: float = Field(..., title="Lon1")
    Lat2: float = Field(..., title="Lat2")
    Lon2: float = Field(..., title="Lon2")


class GICsource(RootModel[Union[GICsource_VoltsAngle, GICsource_ENEELat1Lon1Lat2Lon2]]):
    root: Union[GICsource_VoltsAngle, GICsource_ENEELat1Lon1Lat2Lon2] = Field(..., title="GICsource")

    def dumps_dss(self, edit=False) -> str:
        """
        Dump the DSS object dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            self.dump_dss(output, edit)
            return output.getvalue()

    def dump_dss(self, output: TextIO, edit=False):
        """
        Dump the DSS object to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        GICsource.dict_dump_dss(self.model_dump(exclude_unset=True), output, edit=edit)

    @staticmethod
    def dict_dumps_dss(fields: Dict, edit=False) -> str:
        """
        Dump the DSS object from a dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            GICsource.dict_dump_dss(fields, output, edit)
            return output.getvalue()

    @staticmethod
    def dict_dump_dss(fields: Dict, output: TextIO, edit=False):
        """
        Dump the DSS object from a dict to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        if edit:
            output.write(f'''edit GICsource.{fields['Name']}''')
        else:
            output.write(f'''new GICsource.{fields['Name']}''')

        _Like = fields.get('Like')
        if _Like is not None:
            output.write(f' Like={_quoted(_Like)}')

        _Volts = fields.get('Volts')
        if _Volts is not None:
            output.write(f' Volts={_Volts}')

        _Angle = fields.get('Angle')
        if _Angle is not None:
            output.write(f' Angle={_Angle}')

        _Frequency = fields.get('Frequency')
        if _Frequency is not None:
            output.write(f' Frequency={_Frequency}')

        _Phases = fields.get('Phases')
        if _Phases is not None:
            output.write(f' Phases={_Phases}')

        _EN = fields.get('EN')
        if _EN is not None:
            output.write(f' EN={_EN}')

        _EE = fields.get('EE')
        if _EE is not None:
            output.write(f' EE={_EE}')

        _Lat1 = fields.get('Lat1')
        if _Lat1 is not None:
            output.write(f' Lat1={_Lat1}')

        _Lon1 = fields.get('Lon1')
        if _Lon1 is not None:
            output.write(f' Lon1={_Lon1}')

        _Lat2 = fields.get('Lat2')
        if _Lat2 is not None:
            output.write(f' Lat2={_Lat2}')

        _Lon2 = fields.get('Lon2')
        if _Lon2 is not None:
            output.write(f' Lon2={_Lon2}')

        _Spectrum = fields.get('Spectrum')
        if _Spectrum is not None:
            output.write(f' Spectrum={_quoted(_Spectrum)}')

        _BaseFreq = fields.get('BaseFreq')
        if _BaseFreq is not None:
            output.write(f' BaseFreq={_BaseFreq}')

        _Enabled = fields.get('Enabled')
        if _Enabled is not None:
            output.write(f' Enabled={_Enabled}')

        output.write('\n')

    @model_validator(mode="before")
    def _val_dss_model(self) -> "GICsource":
        _fields_set = set(self.keys())
        # Validate oneOf (spec. sets) based on the `required` lists
        _required_VoltsAngle = _fields_set.issuperset({'Volts'})
        _required_ENEELat1Lon1Lat2Lon2 = _fields_set.issuperset({'EE', 'EN', 'Lat1', 'Lat2', 'Lon1', 'Lon2'})
        num_specs = _required_VoltsAngle + _required_ENEELat1Lon1Lat2Lon2
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



GICsource_ = GICsource


class GICsourceList(RootModel[List[GICsource]]):
    root: List[GICsource]





class GICsourceContainer(RootModel[Union[GICsourceList, JSONFilePath, JSONLinesFilePath]]):
    root: Union[GICsourceList, JSONFilePath, JSONLinesFilePath] = Field(..., title="GICsourceContainer")


    @model_validator(mode="before")
    def _val_dss_model(self) -> "GICsourceContainer":
        try:
            _fields_set = set(self.keys())
        except:
            if isinstance(self, (list, tuple)):
                return self

            raise

        # Validate oneOf (spec. sets) based on the `required` lists
        _required_GICsourceList = "root" in _fields_set and isinstance(self.root, GICsourceList)
        _required_JSONFilePath = _fields_set.issuperset({'JSONFile'})
        _required_JSONLinesFilePath = _fields_set.issuperset({'JSONLinesFile'})
        num_specs = _required_GICsourceList + _required_JSONFilePath + _required_JSONLinesFilePath
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



class AutoTrans_Common(BaseModel):
    Name: Annotated[str, Field(max_length=255, min_length=1)] = Field(..., title="Name")
    Like: Optional[str] = Field(None, title="Like")
    Phases: Optional[PositiveInt] = Field(None, title="Phases")
    Bus: List[BusConnection] = Field(..., title="Bus")
    Conn: Optional[List[AutoTransConnection]] = Field(None, title="Conn")
    kV: List[float] = Field(..., title="kV")
    kVA: Optional[List[float]] = Field(None, title="kVA")
    Tap: Optional[List[float]] = Field(None, title="Tap")
    pctR: Optional[List[float]] = Field(None, title="%R", validation_alias=AliasChoices("pctR", "%R"))
    RDCOhms: Optional[List[float]] = Field(None, title="RDCOhms")
    Core: Optional[CoreType] = Field(None, title="Core")
    Thermal: Optional[float] = Field(None, title="Thermal")
    n: Optional[float] = Field(None, title="n")
    m: Optional[float] = Field(None, title="m")
    FLRise: Optional[float] = Field(None, title="FLRise")
    HSRise: Optional[float] = Field(None, title="HSRise")
    pctLoadLoss: Optional[float] = Field(None, title="%LoadLoss", validation_alias=AliasChoices("pctLoadLoss", "%LoadLoss"))
    pctNoLoadLoss: Optional[float] = Field(None, title="%NoLoadLoss", validation_alias=AliasChoices("pctNoLoadLoss", "%NoLoadLoss"))
    NormHkVA: Optional[float] = Field(None, title="NormHkVA")
    EmergHkVA: Optional[float] = Field(None, title="EmergHkVA")
    Sub: Optional[bool] = Field(None, title="Sub")
    MaxTap: Optional[List[float]] = Field(None, title="MaxTap")
    MinTap: Optional[List[float]] = Field(None, title="MinTap")
    NumTaps: Optional[List[int]] = Field(None, title="NumTaps")
    SubName: Optional[str] = Field(None, title="SubName")
    pctIMag: Optional[float] = Field(None, title="%IMag", validation_alias=AliasChoices("pctIMag", "%IMag"))
    ppm_Antifloat: Optional[float] = Field(None, title="ppm_Antifloat")
    Bank: Optional[str] = Field(None, title="Bank")
    XRConst: Optional[bool] = Field(None, title="XRConst")
    LeadLag: Optional[PhaseSequence] = Field(None, title="LeadLag")
    NormAmps: Optional[float] = Field(None, title="NormAmps")
    EmergAmps: Optional[float] = Field(None, title="EmergAmps")
    FaultRate: Optional[float] = Field(None, title="FaultRate")
    pctPerm: Optional[float] = Field(None, title="pctPerm")
    Repair: Optional[float] = Field(None, title="Repair")
    BaseFreq: Optional[PositiveFloat] = Field(None, title="BaseFreq")
    Enabled: Optional[bool] = Field(None, title="Enabled")

class AutoTrans_XHXXHTXXT(AutoTrans_Common):
    XHX: PositiveFloat = Field(..., title="XHX")
    XHT: Optional[PositiveFloat] = Field(None, title="XHT")
    XXT: Optional[PositiveFloat] = Field(None, title="XXT")

class AutoTrans_XscArray(AutoTrans_Common):
    XSCArray: List[float] = Field(..., title="XSCArray")


class AutoTrans(RootModel[Union[AutoTrans_XHXXHTXXT, AutoTrans_XscArray]]):
    root: Union[AutoTrans_XHXXHTXXT, AutoTrans_XscArray] = Field(..., title="AutoTrans")

    def dumps_dss(self, edit=False) -> str:
        """
        Dump the DSS object dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            self.dump_dss(output, edit)
            return output.getvalue()

    def dump_dss(self, output: TextIO, edit=False):
        """
        Dump the DSS object to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        AutoTrans.dict_dump_dss(self.model_dump(exclude_unset=True), output, edit=edit)

    @staticmethod
    def dict_dumps_dss(fields: Dict, edit=False) -> str:
        """
        Dump the DSS object from a dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            AutoTrans.dict_dump_dss(fields, output, edit)
            return output.getvalue()

    @staticmethod
    def dict_dump_dss(fields: Dict, output: TextIO, edit=False):
        """
        Dump the DSS object from a dict to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        _length_Windings = None

        if edit:
            output.write(f'''edit AutoTrans.{fields['Name']}''')
        else:
            output.write(f'''new AutoTrans.{fields['Name']}''')

        _Like = fields.get('Like')
        if _Like is not None:
            output.write(f' Like={_quoted(_Like)}')

        _Phases = fields.get('Phases')
        if _Phases is not None:
            output.write(f' Phases={_Phases}')

        _Bus = fields.get('Bus')
        _length_Windings = len(_Bus)
        output.write(f' Windings={_length_Windings}')
        output.write(f' Buses=({_quoted_list(_Bus)})')
        _Conn = fields.get('Conn')
        if _Conn is not None:
            if _length_Windings is None:
                _length_Windings = len(_Conn)
                output.write(f' Windings={_length_Windings}')
            elif len(_Conn) != _length_Windings:
                raise ValueError(f'Array length ({len(_Conn)}) for "Conn" does not match expected length ({_length_Windings})')

            output.write(f' Conns=({_quoted_list(_Conn)})')

        _kV = fields.get('kV')
        if _length_Windings is None:
            _length_Windings = len(_kV)
            output.write(f' Windings={_length_Windings}')
        elif len(_kV) != _length_Windings:
            raise ValueError(f'Array length ({len(_kV)}) for "kV" does not match expected length ({_length_Windings})')

        output.write(f' kVs={_as_list(_kV)}')
        _kVA = fields.get('kVA')
        if _kVA is not None:
            if _length_Windings is None:
                _length_Windings = len(_kVA)
                output.write(f' Windings={_length_Windings}')
            elif len(_kVA) != _length_Windings:
                raise ValueError(f'Array length ({len(_kVA)}) for "kVA" does not match expected length ({_length_Windings})')

            output.write(f' kVAs={_as_list(_kVA)}')

        _Tap = fields.get('Tap')
        if _Tap is not None:
            if _length_Windings is None:
                _length_Windings = len(_Tap)
                output.write(f' Windings={_length_Windings}')
            elif len(_Tap) != _length_Windings:
                raise ValueError(f'Array length ({len(_Tap)}) for "Tap" does not match expected length ({_length_Windings})')

            output.write(f' Taps={_as_list(_Tap)}')

        _pctR = fields.get('pctR')
        if _pctR is not None:
            if _length_Windings is None:
                _length_Windings = len(_pctR)
                output.write(f' Windings={_length_Windings}')
            elif len(_pctR) != _length_Windings:
                raise ValueError(f'Array length ({len(_pctR)}) for "pctR" does not match expected length ({_length_Windings})')

            output.write(f' %Rs={_as_list(_pctR)}')

        _Core = fields.get('Core')
        if _Core is not None:
            output.write(f' Core={_quoted(_Core)}')

        _XHX = fields.get('XHX')
        if _XHX is not None:
            output.write(f' XHX={_XHX}')

        _XHT = fields.get('XHT')
        if _XHT is not None:
            output.write(f' XHT={_XHT}')

        _XXT = fields.get('XXT')
        if _XXT is not None:
            output.write(f' XXT={_XXT}')

        _XSCArray = fields.get('XSCArray')
        if _XSCArray is not None:
            output.write(f' XSCArray={_as_list(_XSCArray)}')

        _Thermal = fields.get('Thermal')
        if _Thermal is not None:
            output.write(f' Thermal={_Thermal}')

        _n = fields.get('n')
        if _n is not None:
            output.write(f' n={_n}')

        _m = fields.get('m')
        if _m is not None:
            output.write(f' m={_m}')

        _FLRise = fields.get('FLRise')
        if _FLRise is not None:
            output.write(f' FLRise={_FLRise}')

        _HSRise = fields.get('HSRise')
        if _HSRise is not None:
            output.write(f' HSRise={_HSRise}')

        _pctLoadLoss = fields.get('pctLoadLoss')
        if _pctLoadLoss is not None:
            output.write(f' %LoadLoss={_pctLoadLoss}')

        _pctNoLoadLoss = fields.get('pctNoLoadLoss')
        if _pctNoLoadLoss is not None:
            output.write(f' %NoLoadLoss={_pctNoLoadLoss}')

        _NormHkVA = fields.get('NormHkVA')
        if _NormHkVA is not None:
            output.write(f' NormHkVA={_NormHkVA}')

        _EmergHkVA = fields.get('EmergHkVA')
        if _EmergHkVA is not None:
            output.write(f' EmergHkVA={_EmergHkVA}')

        _Sub = fields.get('Sub')
        if _Sub is not None:
            output.write(f' Sub={_Sub}')

        _SubName = fields.get('SubName')
        if _SubName is not None:
            output.write(f' SubName={_quoted(_SubName)}')

        _pctIMag = fields.get('pctIMag')
        if _pctIMag is not None:
            output.write(f' %IMag={_pctIMag}')

        _ppm_Antifloat = fields.get('ppm_Antifloat')
        if _ppm_Antifloat is not None:
            output.write(f' ppm_Antifloat={_ppm_Antifloat}')

        _Bank = fields.get('Bank')
        if _Bank is not None:
            output.write(f' Bank={_quoted(_Bank)}')

        _XRConst = fields.get('XRConst')
        if _XRConst is not None:
            output.write(f' XRConst={_XRConst}')

        _LeadLag = fields.get('LeadLag')
        if _LeadLag is not None:
            output.write(f' LeadLag={_quoted(_LeadLag)}')

        _NormAmps = fields.get('NormAmps')
        if _NormAmps is not None:
            output.write(f' NormAmps={_NormAmps}')

        _EmergAmps = fields.get('EmergAmps')
        if _EmergAmps is not None:
            output.write(f' EmergAmps={_EmergAmps}')

        _FaultRate = fields.get('FaultRate')
        if _FaultRate is not None:
            output.write(f' FaultRate={_FaultRate}')

        _pctPerm = fields.get('pctPerm')
        if _pctPerm is not None:
            output.write(f' pctPerm={_pctPerm}')

        _Repair = fields.get('Repair')
        if _Repair is not None:
            output.write(f' Repair={_Repair}')

        _BaseFreq = fields.get('BaseFreq')
        if _BaseFreq is not None:
            output.write(f' BaseFreq={_BaseFreq}')

        _Enabled = fields.get('Enabled')
        if _Enabled is not None:
            output.write(f' Enabled={_Enabled}')

        _needs_Wdg = False
        _RDCOhms = fields.get('RDCOhms')
        if _RDCOhms is not None:
            if len(_RDCOhms) != _length_Windings:
                raise ValueError(f'Array length ({len(_RDCOhms)}) for "RDCOhms" does not match expected length ({_length_Windings})')
            _needs_Wdg = True
        _MaxTap = fields.get('MaxTap')
        if _MaxTap is not None:
            if len(_MaxTap) != _length_Windings:
                raise ValueError(f'Array length ({len(_MaxTap)}) for "MaxTap" does not match expected length ({_length_Windings})')
            _needs_Wdg = True
        _MinTap = fields.get('MinTap')
        if _MinTap is not None:
            if len(_MinTap) != _length_Windings:
                raise ValueError(f'Array length ({len(_MinTap)}) for "MinTap" does not match expected length ({_length_Windings})')
            _needs_Wdg = True
        _NumTaps = fields.get('NumTaps')
        if _NumTaps is not None:
            if len(_NumTaps) != _length_Windings:
                raise ValueError(f'Array length ({len(_NumTaps)}) for "NumTaps" does not match expected length ({_length_Windings})')
            _needs_Wdg = True
        if _length_Windings is not None and _needs_Wdg:
            for _Wdg in range(_length_Windings):
                output.write(f" Wdg={_Wdg + 1}")
                _RDCOhms = fields.get('RDCOhms')
                if _RDCOhms is not None:
                    output.write(f" RDCOhms={_RDCOhms[_Wdg]}")
                _MaxTap = fields.get('MaxTap')
                if _MaxTap is not None:
                    output.write(f" MaxTap={_MaxTap[_Wdg]}")
                _MinTap = fields.get('MinTap')
                if _MinTap is not None:
                    output.write(f" MinTap={_MinTap[_Wdg]}")
                _NumTaps = fields.get('NumTaps')
                if _NumTaps is not None:
                    output.write(f" NumTaps={_NumTaps[_Wdg]}")
        output.write('\n')

    @model_validator(mode="before")
    def _val_dss_model(self) -> "AutoTrans":
        _fields_set = set(self.keys())
        # Validate oneOf (spec. sets) based on the `required` lists
        _required_XHXXHTXXT = _fields_set.issuperset({'XHX'})
        _required_XscArray = _fields_set.issuperset({'XSCArray'})
        num_specs = _required_XHXXHTXXT + _required_XscArray
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



AutoTrans_ = AutoTrans


class AutoTransList(RootModel[List[AutoTrans]]):
    root: List[AutoTrans]





class AutoTransContainer(RootModel[Union[AutoTransList, JSONFilePath, JSONLinesFilePath]]):
    root: Union[AutoTransList, JSONFilePath, JSONLinesFilePath] = Field(..., title="AutoTransContainer")


    @model_validator(mode="before")
    def _val_dss_model(self) -> "AutoTransContainer":
        try:
            _fields_set = set(self.keys())
        except:
            if isinstance(self, (list, tuple)):
                return self

            raise

        # Validate oneOf (spec. sets) based on the `required` lists
        _required_AutoTransList = "root" in _fields_set and isinstance(self.root, AutoTransList)
        _required_JSONFilePath = _fields_set.issuperset({'JSONFile'})
        _required_JSONLinesFilePath = _fields_set.issuperset({'JSONLinesFile'})
        num_specs = _required_AutoTransList + _required_JSONFilePath + _required_JSONLinesFilePath
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



class InvControl(BaseModel):
    Name: Annotated[str, Field(max_length=255, min_length=1)] = Field(..., title="Name")
    Like: Optional[str] = Field(None, title="Like")
    DERList: Optional[StringArrayOrFilePath] = Field(None, title="DERList")
    Mode: Optional[InvControlControlMode] = Field(None, title="Mode")
    CombiMode: Optional[InvControlCombiMode] = Field(None, title="CombiMode")
    VVC_Curve1: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="VVC_Curve1")
    Hysteresis_Offset: Optional[float] = Field(None, title="Hysteresis_Offset")
    Voltage_CurveX_Ref: Optional[InvControlVoltageCurveXRef] = Field(None, title="Voltage_CurveX_Ref")
    AvgWindowLen: Optional[int] = Field(None, title="AvgWindowLen")
    VoltWatt_Curve: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="VoltWatt_Curve")
    DbVMin: Optional[float] = Field(None, title="DbVMin")
    DbVMax: Optional[float] = Field(None, title="DbVMax")
    ArGraLowV: Optional[float] = Field(None, title="ArGraLowV")
    ArGraHiV: Optional[float] = Field(None, title="ArGraHiV")
    DynReacAvgWindowLen: Optional[int] = Field(None, title="DynReacAvgWindowLen")
    DeltaQ_Factor: Optional[float] = Field(None, title="DeltaQ_Factor")
    VoltageChangeTolerance: Optional[float] = Field(None, title="VoltageChangeTolerance")
    VarChangeTolerance: Optional[float] = Field(None, title="VarChangeTolerance")
    VoltWattYAxis: Optional[InvControlVoltWattYAxis] = Field(None, title="VoltWattYAxis")
    RateOfChangeMode: Optional[InvControlRateOfChangeMode] = Field(None, title="RateOfChangeMode")
    LPFTau: Optional[float] = Field(None, title="LPFTau")
    RiseFallLimit: Optional[float] = Field(None, title="RiseFallLimit")
    DeltaP_Factor: Optional[float] = Field(None, title="DeltaP_Factor")
    EventLog: Optional[bool] = Field(None, title="EventLog")
    RefReactivePower: Optional[InvControlReactivePowerReference] = Field(None, title="RefReactivePower")
    ActivePChangeTolerance: Optional[float] = Field(None, title="ActivePChangeTolerance")
    MonVoltageCalc: Optional[MonitoredPhase] = Field(None, title="MonVoltageCalc")
    MonBus: Optional[StringArrayOrFilePath] = Field(None, title="MonBus")
    MonBusesVBase: Optional[List[float]] = Field(None, title="MonBusesVBase")
    VoltWattCH_Curve: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="VoltWattCH_Curve")
    WattPF_Curve: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="WattPF_Curve")
    WattVar_Curve: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="WattVar_Curve")
    VSetPoint: Optional[float] = Field(None, title="VSetPoint")
    ControlModel: Optional[InvControlControlModel] = Field(None, title="ControlModel")
    BaseFreq: Optional[PositiveFloat] = Field(None, title="BaseFreq")
    Enabled: Optional[bool] = Field(None, title="Enabled")

    def dumps_dss(self, edit=False) -> str:
        """
        Dump the DSS object dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            self.dump_dss(output, edit)
            return output.getvalue()

    def dump_dss(self, output: TextIO, edit=False):
        """
        Dump the DSS object to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        InvControl.dict_dump_dss(self.model_dump(exclude_unset=True), output, edit=edit)

    @staticmethod
    def dict_dumps_dss(fields: Dict, edit=False) -> str:
        """
        Dump the DSS object from a dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            InvControl.dict_dump_dss(fields, output, edit)
            return output.getvalue()

    @staticmethod
    def dict_dump_dss(fields: Dict, output: TextIO, edit=False):
        """
        Dump the DSS object from a dict to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        if edit:
            output.write(f'''edit InvControl.{fields['Name']}''')
        else:
            output.write(f'''new InvControl.{fields['Name']}''')

        _Like = fields.get('Like')
        if _Like is not None:
            output.write(f' Like={_quoted(_Like)}')

        _DERList = fields.get('DERList')
        if _DERList is not None:
            output.write(f' DERList=({_filepath_stringlist(_DERList)})')

        _Mode = fields.get('Mode')
        if _Mode is not None:
            output.write(f' Mode={_quoted(_Mode)}')

        _CombiMode = fields.get('CombiMode')
        if _CombiMode is not None:
            output.write(f' CombiMode={_quoted(_CombiMode)}')

        _VVC_Curve1 = fields.get('VVC_Curve1')
        if _VVC_Curve1 is not None:
            output.write(f' VVC_Curve1={_quoted(_VVC_Curve1)}')

        _Hysteresis_Offset = fields.get('Hysteresis_Offset')
        if _Hysteresis_Offset is not None:
            output.write(f' Hysteresis_Offset={_Hysteresis_Offset}')

        _Voltage_CurveX_Ref = fields.get('Voltage_CurveX_Ref')
        if _Voltage_CurveX_Ref is not None:
            output.write(f' Voltage_CurveX_Ref={_quoted(_Voltage_CurveX_Ref)}')

        _AvgWindowLen = fields.get('AvgWindowLen')
        if _AvgWindowLen is not None:
            output.write(f' AvgWindowLen={_AvgWindowLen}')

        _VoltWatt_Curve = fields.get('VoltWatt_Curve')
        if _VoltWatt_Curve is not None:
            output.write(f' VoltWatt_Curve={_quoted(_VoltWatt_Curve)}')

        _DbVMin = fields.get('DbVMin')
        if _DbVMin is not None:
            output.write(f' DbVMin={_DbVMin}')

        _DbVMax = fields.get('DbVMax')
        if _DbVMax is not None:
            output.write(f' DbVMax={_DbVMax}')

        _ArGraLowV = fields.get('ArGraLowV')
        if _ArGraLowV is not None:
            output.write(f' ArGraLowV={_ArGraLowV}')

        _ArGraHiV = fields.get('ArGraHiV')
        if _ArGraHiV is not None:
            output.write(f' ArGraHiV={_ArGraHiV}')

        _DynReacAvgWindowLen = fields.get('DynReacAvgWindowLen')
        if _DynReacAvgWindowLen is not None:
            output.write(f' DynReacAvgWindowLen={_DynReacAvgWindowLen}')

        _DeltaQ_Factor = fields.get('DeltaQ_Factor')
        if _DeltaQ_Factor is not None:
            output.write(f' DeltaQ_Factor={_DeltaQ_Factor}')

        _VoltageChangeTolerance = fields.get('VoltageChangeTolerance')
        if _VoltageChangeTolerance is not None:
            output.write(f' VoltageChangeTolerance={_VoltageChangeTolerance}')

        _VarChangeTolerance = fields.get('VarChangeTolerance')
        if _VarChangeTolerance is not None:
            output.write(f' VarChangeTolerance={_VarChangeTolerance}')

        _VoltWattYAxis = fields.get('VoltWattYAxis')
        if _VoltWattYAxis is not None:
            output.write(f' VoltWattYAxis={_quoted(_VoltWattYAxis)}')

        _RateOfChangeMode = fields.get('RateOfChangeMode')
        if _RateOfChangeMode is not None:
            output.write(f' RateOfChangeMode={_quoted(_RateOfChangeMode)}')

        _LPFTau = fields.get('LPFTau')
        if _LPFTau is not None:
            output.write(f' LPFTau={_LPFTau}')

        _RiseFallLimit = fields.get('RiseFallLimit')
        if _RiseFallLimit is not None:
            output.write(f' RiseFallLimit={_RiseFallLimit}')

        _DeltaP_Factor = fields.get('DeltaP_Factor')
        if _DeltaP_Factor is not None:
            output.write(f' DeltaP_Factor={_DeltaP_Factor}')

        _EventLog = fields.get('EventLog')
        if _EventLog is not None:
            output.write(f' EventLog={_EventLog}')

        _RefReactivePower = fields.get('RefReactivePower')
        if _RefReactivePower is not None:
            output.write(f' RefReactivePower={_quoted(_RefReactivePower)}')

        _ActivePChangeTolerance = fields.get('ActivePChangeTolerance')
        if _ActivePChangeTolerance is not None:
            output.write(f' ActivePChangeTolerance={_ActivePChangeTolerance}')

        _MonVoltageCalc = fields.get('MonVoltageCalc')
        if _MonVoltageCalc is not None:
            output.write(f' MonVoltageCalc={_MonVoltageCalc}')

        _MonBus = fields.get('MonBus')
        if _MonBus is not None:
            output.write(f' MonBus=({_filepath_stringlist(_MonBus)})')

        _MonBusesVBase = fields.get('MonBusesVBase')
        if _MonBusesVBase is not None:
            output.write(f' MonBusesVBase={_as_list(_MonBusesVBase)}')

        _VoltWattCH_Curve = fields.get('VoltWattCH_Curve')
        if _VoltWattCH_Curve is not None:
            output.write(f' VoltWattCH_Curve={_quoted(_VoltWattCH_Curve)}')

        _WattPF_Curve = fields.get('WattPF_Curve')
        if _WattPF_Curve is not None:
            output.write(f' WattPF_Curve={_quoted(_WattPF_Curve)}')

        _WattVar_Curve = fields.get('WattVar_Curve')
        if _WattVar_Curve is not None:
            output.write(f' WattVar_Curve={_quoted(_WattVar_Curve)}')

        _VSetPoint = fields.get('VSetPoint')
        if _VSetPoint is not None:
            output.write(f' VSetPoint={_VSetPoint}')

        _ControlModel = fields.get('ControlModel')
        if _ControlModel is not None:
            output.write(f' ControlModel={_ControlModel}')

        _BaseFreq = fields.get('BaseFreq')
        if _BaseFreq is not None:
            output.write(f' BaseFreq={_BaseFreq}')

        _Enabled = fields.get('Enabled')
        if _Enabled is not None:
            output.write(f' Enabled={_Enabled}')

        output.write('\n')


InvControl_ = InvControl


class InvControlList(RootModel[List[InvControl]]):
    root: List[InvControl]





class InvControlContainer(RootModel[Union[InvControlList, JSONFilePath, JSONLinesFilePath]]):
    root: Union[InvControlList, JSONFilePath, JSONLinesFilePath] = Field(..., title="InvControlContainer")


    @model_validator(mode="before")
    def _val_dss_model(self) -> "InvControlContainer":
        try:
            _fields_set = set(self.keys())
        except:
            if isinstance(self, (list, tuple)):
                return self

            raise

        # Validate oneOf (spec. sets) based on the `required` lists
        _required_InvControlList = "root" in _fields_set and isinstance(self.root, InvControlList)
        _required_JSONFilePath = _fields_set.issuperset({'JSONFile'})
        _required_JSONLinesFilePath = _fields_set.issuperset({'JSONLinesFile'})
        num_specs = _required_InvControlList + _required_JSONFilePath + _required_JSONLinesFilePath
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



class ExpControl(BaseModel):
    Name: Annotated[str, Field(max_length=255, min_length=1)] = Field(..., title="Name")
    Like: Optional[str] = Field(None, title="Like")
    PVSystemList: Optional[StringArrayOrFilePath] = Field(None, title="PVSystemList")
    VReg: Optional[Annotated[float, Field(ge=0)]] = Field(None, title="VReg")
    Slope: Optional[PositiveFloat] = Field(None, title="Slope")
    VRegTau: Optional[Annotated[float, Field(ge=0)]] = Field(None, title="VRegTau")
    QBias: Optional[float] = Field(None, title="QBias")
    VRegMin: Optional[Annotated[float, Field(ge=0)]] = Field(None, title="VRegMin")
    VRegMax: Optional[Annotated[float, Field(ge=0)]] = Field(None, title="VRegMax")
    QMaxLead: Optional[PositiveFloat] = Field(None, title="QMaxLead")
    QMaxLag: Optional[PositiveFloat] = Field(None, title="QMaxLag")
    EventLog: Optional[bool] = Field(None, title="EventLog")
    DeltaQ_Factor: Optional[float] = Field(None, title="DeltaQ_Factor")
    PreferQ: Optional[bool] = Field(None, title="PreferQ")
    TResponse: Optional[PositiveFloat] = Field(None, title="TResponse")
    DERList: Optional[StringArrayOrFilePath] = Field(None, title="DERList")
    BaseFreq: Optional[PositiveFloat] = Field(None, title="BaseFreq")
    Enabled: Optional[bool] = Field(None, title="Enabled")

    def dumps_dss(self, edit=False) -> str:
        """
        Dump the DSS object dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            self.dump_dss(output, edit)
            return output.getvalue()

    def dump_dss(self, output: TextIO, edit=False):
        """
        Dump the DSS object to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        ExpControl.dict_dump_dss(self.model_dump(exclude_unset=True), output, edit=edit)

    @staticmethod
    def dict_dumps_dss(fields: Dict, edit=False) -> str:
        """
        Dump the DSS object from a dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            ExpControl.dict_dump_dss(fields, output, edit)
            return output.getvalue()

    @staticmethod
    def dict_dump_dss(fields: Dict, output: TextIO, edit=False):
        """
        Dump the DSS object from a dict to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        if edit:
            output.write(f'''edit ExpControl.{fields['Name']}''')
        else:
            output.write(f'''new ExpControl.{fields['Name']}''')

        _Like = fields.get('Like')
        if _Like is not None:
            output.write(f' Like={_quoted(_Like)}')

        _PVSystemList = fields.get('PVSystemList')
        if _PVSystemList is not None:
            output.write(f' PVSystemList=({_filepath_stringlist(_PVSystemList)})')

        _VReg = fields.get('VReg')
        if _VReg is not None:
            output.write(f' VReg={_VReg}')

        _Slope = fields.get('Slope')
        if _Slope is not None:
            output.write(f' Slope={_Slope}')

        _VRegTau = fields.get('VRegTau')
        if _VRegTau is not None:
            output.write(f' VRegTau={_VRegTau}')

        _QBias = fields.get('QBias')
        if _QBias is not None:
            output.write(f' QBias={_QBias}')

        _VRegMin = fields.get('VRegMin')
        if _VRegMin is not None:
            output.write(f' VRegMin={_VRegMin}')

        _VRegMax = fields.get('VRegMax')
        if _VRegMax is not None:
            output.write(f' VRegMax={_VRegMax}')

        _QMaxLead = fields.get('QMaxLead')
        if _QMaxLead is not None:
            output.write(f' QMaxLead={_QMaxLead}')

        _QMaxLag = fields.get('QMaxLag')
        if _QMaxLag is not None:
            output.write(f' QMaxLag={_QMaxLag}')

        _EventLog = fields.get('EventLog')
        if _EventLog is not None:
            output.write(f' EventLog={_EventLog}')

        _DeltaQ_Factor = fields.get('DeltaQ_Factor')
        if _DeltaQ_Factor is not None:
            output.write(f' DeltaQ_Factor={_DeltaQ_Factor}')

        _PreferQ = fields.get('PreferQ')
        if _PreferQ is not None:
            output.write(f' PreferQ={_PreferQ}')

        _TResponse = fields.get('TResponse')
        if _TResponse is not None:
            output.write(f' TResponse={_TResponse}')

        _DERList = fields.get('DERList')
        if _DERList is not None:
            output.write(f' DERList=({_filepath_stringlist(_DERList)})')

        _BaseFreq = fields.get('BaseFreq')
        if _BaseFreq is not None:
            output.write(f' BaseFreq={_BaseFreq}')

        _Enabled = fields.get('Enabled')
        if _Enabled is not None:
            output.write(f' Enabled={_Enabled}')

        output.write('\n')


ExpControl_ = ExpControl


class ExpControlList(RootModel[List[ExpControl]]):
    root: List[ExpControl]





class ExpControlContainer(RootModel[Union[ExpControlList, JSONFilePath, JSONLinesFilePath]]):
    root: Union[ExpControlList, JSONFilePath, JSONLinesFilePath] = Field(..., title="ExpControlContainer")


    @model_validator(mode="before")
    def _val_dss_model(self) -> "ExpControlContainer":
        try:
            _fields_set = set(self.keys())
        except:
            if isinstance(self, (list, tuple)):
                return self

            raise

        # Validate oneOf (spec. sets) based on the `required` lists
        _required_ExpControlList = "root" in _fields_set and isinstance(self.root, ExpControlList)
        _required_JSONFilePath = _fields_set.issuperset({'JSONFile'})
        _required_JSONLinesFilePath = _fields_set.issuperset({'JSONLinesFile'})
        num_specs = _required_ExpControlList + _required_JSONFilePath + _required_JSONLinesFilePath
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



class GICLine_Common(BaseModel):
    Name: Annotated[str, Field(max_length=255, min_length=1)] = Field(..., title="Name")
    Like: Optional[str] = Field(None, title="Like")
    Bus1: BusConnection = Field(..., title="Bus1")
    Bus2: Optional[BusConnection] = Field(None, title="Bus2")
    Frequency: Optional[PositiveFloat] = Field(None, title="Frequency")
    Phases: Optional[PositiveInt] = Field(None, title="Phases")
    R: Optional[float] = Field(None, title="R")
    X: Optional[float] = Field(None, title="X")
    C: Optional[float] = Field(None, title="C")
    Enabled: Optional[bool] = Field(None, title="Enabled")

class GICLine_VoltsAngle(GICLine_Common):
    Volts: float = Field(..., title="Volts")
    Angle: Optional[float] = Field(None, title="Angle")

class GICLine_ENEELat1Lon1Lat2Lon2(GICLine_Common):
    EN: float = Field(..., title="EN")
    EE: float = Field(..., title="EE")
    Lat1: float = Field(..., title="Lat1")
    Lon1: float = Field(..., title="Lon1")
    Lat2: float = Field(..., title="Lat2")
    Lon2: float = Field(..., title="Lon2")


class GICLine(RootModel[Union[GICLine_VoltsAngle, GICLine_ENEELat1Lon1Lat2Lon2]]):
    root: Union[GICLine_VoltsAngle, GICLine_ENEELat1Lon1Lat2Lon2] = Field(..., title="GICLine")

    def dumps_dss(self, edit=False) -> str:
        """
        Dump the DSS object dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            self.dump_dss(output, edit)
            return output.getvalue()

    def dump_dss(self, output: TextIO, edit=False):
        """
        Dump the DSS object to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        GICLine.dict_dump_dss(self.model_dump(exclude_unset=True), output, edit=edit)

    @staticmethod
    def dict_dumps_dss(fields: Dict, edit=False) -> str:
        """
        Dump the DSS object from a dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            GICLine.dict_dump_dss(fields, output, edit)
            return output.getvalue()

    @staticmethod
    def dict_dump_dss(fields: Dict, output: TextIO, edit=False):
        """
        Dump the DSS object from a dict to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        if edit:
            output.write(f'''edit GICLine.{fields['Name']}''')
        else:
            output.write(f'''new GICLine.{fields['Name']}''')

        _Like = fields.get('Like')
        if _Like is not None:
            output.write(f' Like={_quoted(_Like)}')

        _Bus1 = fields.get('Bus1')
        output.write(f' Bus1={_quoted(_Bus1)}')
        _Bus2 = fields.get('Bus2')
        if _Bus2 is not None:
            output.write(f' Bus2={_quoted(_Bus2)}')

        _Volts = fields.get('Volts')
        if _Volts is not None:
            output.write(f' Volts={_Volts}')

        _Angle = fields.get('Angle')
        if _Angle is not None:
            output.write(f' Angle={_Angle}')

        _Frequency = fields.get('Frequency')
        if _Frequency is not None:
            output.write(f' Frequency={_Frequency}')

        _Phases = fields.get('Phases')
        if _Phases is not None:
            output.write(f' Phases={_Phases}')

        _R = fields.get('R')
        if _R is not None:
            output.write(f' R={_R}')

        _X = fields.get('X')
        if _X is not None:
            output.write(f' X={_X}')

        _C = fields.get('C')
        if _C is not None:
            output.write(f' C={_C}')

        _EN = fields.get('EN')
        if _EN is not None:
            output.write(f' EN={_EN}')

        _EE = fields.get('EE')
        if _EE is not None:
            output.write(f' EE={_EE}')

        _Lat1 = fields.get('Lat1')
        if _Lat1 is not None:
            output.write(f' Lat1={_Lat1}')

        _Lon1 = fields.get('Lon1')
        if _Lon1 is not None:
            output.write(f' Lon1={_Lon1}')

        _Lat2 = fields.get('Lat2')
        if _Lat2 is not None:
            output.write(f' Lat2={_Lat2}')

        _Lon2 = fields.get('Lon2')
        if _Lon2 is not None:
            output.write(f' Lon2={_Lon2}')

        _Enabled = fields.get('Enabled')
        if _Enabled is not None:
            output.write(f' Enabled={_Enabled}')

        output.write('\n')

    @model_validator(mode="before")
    def _val_dss_model(self) -> "GICLine":
        _fields_set = set(self.keys())
        # Validate oneOf (spec. sets) based on the `required` lists
        _required_VoltsAngle = _fields_set.issuperset({'Volts'})
        _required_ENEELat1Lon1Lat2Lon2 = _fields_set.issuperset({'EE', 'EN', 'Lat1', 'Lat2', 'Lon1', 'Lon2'})
        num_specs = _required_VoltsAngle + _required_ENEELat1Lon1Lat2Lon2
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



GICLine_ = GICLine


class GICLineList(RootModel[List[GICLine]]):
    root: List[GICLine]





class GICLineContainer(RootModel[Union[GICLineList, JSONFilePath, JSONLinesFilePath]]):
    root: Union[GICLineList, JSONFilePath, JSONLinesFilePath] = Field(..., title="GICLineContainer")


    @model_validator(mode="before")
    def _val_dss_model(self) -> "GICLineContainer":
        try:
            _fields_set = set(self.keys())
        except:
            if isinstance(self, (list, tuple)):
                return self

            raise

        # Validate oneOf (spec. sets) based on the `required` lists
        _required_GICLineList = "root" in _fields_set and isinstance(self.root, GICLineList)
        _required_JSONFilePath = _fields_set.issuperset({'JSONFile'})
        _required_JSONLinesFilePath = _fields_set.issuperset({'JSONLinesFile'})
        num_specs = _required_GICLineList + _required_JSONFilePath + _required_JSONLinesFilePath
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



class GICTransformer_Common(BaseModel):
    Name: Annotated[str, Field(max_length=255, min_length=1)] = Field(..., title="Name")
    Like: Optional[str] = Field(None, title="Like")
    BusH: Optional[BusConnection] = Field(None, title="BusH")
    BusNH: Optional[BusConnection] = Field(None, title="BusNH")
    BusX: Optional[BusConnection] = Field(None, title="BusX")
    BusNX: Optional[BusConnection] = Field(None, title="BusNX")
    Phases: Optional[PositiveInt] = Field(None, title="Phases")
    Type: Optional[GICTransformerType] = Field(None, title="Type")
    kVLL1: Optional[float] = Field(None, title="kVLL1")
    kVLL2: Optional[float] = Field(None, title="kVLL2")
    MVA: Optional[float] = Field(None, title="MVA")
    VarCurve: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="VarCurve")
    K: Optional[float] = Field(None, title="K")
    NormAmps: Optional[float] = Field(None, title="NormAmps")
    EmergAmps: Optional[float] = Field(None, title="EmergAmps")
    FaultRate: Optional[float] = Field(None, title="FaultRate")
    pctPerm: Optional[float] = Field(None, title="pctPerm")
    Repair: Optional[float] = Field(None, title="Repair")
    BaseFreq: Optional[PositiveFloat] = Field(None, title="BaseFreq")
    Enabled: Optional[bool] = Field(None, title="Enabled")

class GICTransformer_R1R2(GICTransformer_Common):
    R1: float = Field(..., title="R1")
    R2: Optional[float] = Field(None, title="R2")

class GICTransformer_pctR1pctR2(GICTransformer_Common):
    pctR1: float = Field(..., title="%R1", validation_alias=AliasChoices("pctR1", "%R1"))
    pctR2: Optional[float] = Field(None, title="%R2", validation_alias=AliasChoices("pctR2", "%R2"))


class GICTransformer(RootModel[Union[GICTransformer_R1R2, GICTransformer_pctR1pctR2]]):
    root: Union[GICTransformer_R1R2, GICTransformer_pctR1pctR2] = Field(..., title="GICTransformer")

    def dumps_dss(self, edit=False) -> str:
        """
        Dump the DSS object dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            self.dump_dss(output, edit)
            return output.getvalue()

    def dump_dss(self, output: TextIO, edit=False):
        """
        Dump the DSS object to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        GICTransformer.dict_dump_dss(self.model_dump(exclude_unset=True), output, edit=edit)

    @staticmethod
    def dict_dumps_dss(fields: Dict, edit=False) -> str:
        """
        Dump the DSS object from a dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            GICTransformer.dict_dump_dss(fields, output, edit)
            return output.getvalue()

    @staticmethod
    def dict_dump_dss(fields: Dict, output: TextIO, edit=False):
        """
        Dump the DSS object from a dict to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        if edit:
            output.write(f'''edit GICTransformer.{fields['Name']}''')
        else:
            output.write(f'''new GICTransformer.{fields['Name']}''')

        _Like = fields.get('Like')
        if _Like is not None:
            output.write(f' Like={_quoted(_Like)}')

        _BusH = fields.get('BusH')
        if _BusH is not None:
            output.write(f' BusH={_quoted(_BusH)}')

        _BusNH = fields.get('BusNH')
        if _BusNH is not None:
            output.write(f' BusNH={_quoted(_BusNH)}')

        _BusX = fields.get('BusX')
        if _BusX is not None:
            output.write(f' BusX={_quoted(_BusX)}')

        _BusNX = fields.get('BusNX')
        if _BusNX is not None:
            output.write(f' BusNX={_quoted(_BusNX)}')

        _Phases = fields.get('Phases')
        if _Phases is not None:
            output.write(f' Phases={_Phases}')

        _Type = fields.get('Type')
        if _Type is not None:
            output.write(f' Type={_quoted(_Type)}')

        _R1 = fields.get('R1')
        if _R1 is not None:
            output.write(f' R1={_R1}')

        _R2 = fields.get('R2')
        if _R2 is not None:
            output.write(f' R2={_R2}')

        _kVLL1 = fields.get('kVLL1')
        if _kVLL1 is not None:
            output.write(f' kVLL1={_kVLL1}')

        _kVLL2 = fields.get('kVLL2')
        if _kVLL2 is not None:
            output.write(f' kVLL2={_kVLL2}')

        _MVA = fields.get('MVA')
        if _MVA is not None:
            output.write(f' MVA={_MVA}')

        _VarCurve = fields.get('VarCurve')
        if _VarCurve is not None:
            output.write(f' VarCurve={_quoted(_VarCurve)}')

        _pctR1 = fields.get('pctR1')
        if _pctR1 is not None:
            output.write(f' %R1={_pctR1}')

        _pctR2 = fields.get('pctR2')
        if _pctR2 is not None:
            output.write(f' %R2={_pctR2}')

        _K = fields.get('K')
        if _K is not None:
            output.write(f' K={_K}')

        _NormAmps = fields.get('NormAmps')
        if _NormAmps is not None:
            output.write(f' NormAmps={_NormAmps}')

        _EmergAmps = fields.get('EmergAmps')
        if _EmergAmps is not None:
            output.write(f' EmergAmps={_EmergAmps}')

        _FaultRate = fields.get('FaultRate')
        if _FaultRate is not None:
            output.write(f' FaultRate={_FaultRate}')

        _pctPerm = fields.get('pctPerm')
        if _pctPerm is not None:
            output.write(f' pctPerm={_pctPerm}')

        _Repair = fields.get('Repair')
        if _Repair is not None:
            output.write(f' Repair={_Repair}')

        _BaseFreq = fields.get('BaseFreq')
        if _BaseFreq is not None:
            output.write(f' BaseFreq={_BaseFreq}')

        _Enabled = fields.get('Enabled')
        if _Enabled is not None:
            output.write(f' Enabled={_Enabled}')

        output.write('\n')

    @model_validator(mode="before")
    def _val_dss_model(self) -> "GICTransformer":
        _fields_set = set(self.keys())
        # Validate oneOf (spec. sets) based on the `required` lists
        _required_R1R2 = _fields_set.issuperset({'R1'})
        _required_pctR1pctR2 = _fields_set.issuperset({'pctR1'})
        num_specs = _required_R1R2 + _required_pctR1pctR2
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



GICTransformer_ = GICTransformer


class GICTransformerList(RootModel[List[GICTransformer]]):
    root: List[GICTransformer]





class GICTransformerContainer(RootModel[Union[GICTransformerList, JSONFilePath, JSONLinesFilePath]]):
    root: Union[GICTransformerList, JSONFilePath, JSONLinesFilePath] = Field(..., title="GICTransformerContainer")


    @model_validator(mode="before")
    def _val_dss_model(self) -> "GICTransformerContainer":
        try:
            _fields_set = set(self.keys())
        except:
            if isinstance(self, (list, tuple)):
                return self

            raise

        # Validate oneOf (spec. sets) based on the `required` lists
        _required_GICTransformerList = "root" in _fields_set and isinstance(self.root, GICTransformerList)
        _required_JSONFilePath = _fields_set.issuperset({'JSONFile'})
        _required_JSONLinesFilePath = _fields_set.issuperset({'JSONLinesFile'})
        num_specs = _required_GICTransformerList + _required_JSONFilePath + _required_JSONLinesFilePath
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



class VSConverter(BaseModel):
    Name: Annotated[str, Field(max_length=255, min_length=1)] = Field(..., title="Name")
    Like: Optional[str] = Field(None, title="Like")
    Phases: Optional[PositiveInt] = Field(None, title="Phases")
    Bus1: BusConnection = Field(..., title="Bus1")
    kVAC: Optional[float] = Field(None, title="kVAC")
    kVDC: Optional[float] = Field(None, title="kVDC")
    kW: Optional[float] = Field(None, title="kW")
    NDC: Optional[int] = Field(None, title="NDC")
    RAC: Optional[float] = Field(None, title="RAC")
    XAC: Optional[float] = Field(None, title="XAC")
    M0: Optional[float] = Field(None, title="M0")
    d0: Optional[float] = Field(None, title="d0")
    MMin: Optional[float] = Field(None, title="MMin")
    MMax: Optional[float] = Field(None, title="MMax")
    IACMax: Optional[float] = Field(None, title="IACMax")
    IDCMax: Optional[float] = Field(None, title="IDCMax")
    VACRef: Optional[float] = Field(None, title="VACRef")
    PACRef: Optional[float] = Field(None, title="PACRef")
    QACRef: Optional[float] = Field(None, title="QACRef")
    VDCRef: Optional[float] = Field(None, title="VDCRef")
    VSCMode: Optional[VSConverterControlMode] = Field(None, title="VSCMode")
    Spectrum: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="Spectrum")
    BaseFreq: Optional[PositiveFloat] = Field(None, title="BaseFreq")
    Enabled: Optional[bool] = Field(None, title="Enabled")

    def dumps_dss(self, edit=False) -> str:
        """
        Dump the DSS object dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            self.dump_dss(output, edit)
            return output.getvalue()

    def dump_dss(self, output: TextIO, edit=False):
        """
        Dump the DSS object to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        VSConverter.dict_dump_dss(self.model_dump(exclude_unset=True), output, edit=edit)

    @staticmethod
    def dict_dumps_dss(fields: Dict, edit=False) -> str:
        """
        Dump the DSS object from a dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            VSConverter.dict_dump_dss(fields, output, edit)
            return output.getvalue()

    @staticmethod
    def dict_dump_dss(fields: Dict, output: TextIO, edit=False):
        """
        Dump the DSS object from a dict to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        if edit:
            output.write(f'''edit VSConverter.{fields['Name']}''')
        else:
            output.write(f'''new VSConverter.{fields['Name']}''')

        _Like = fields.get('Like')
        if _Like is not None:
            output.write(f' Like={_quoted(_Like)}')

        _Phases = fields.get('Phases')
        if _Phases is not None:
            output.write(f' Phases={_Phases}')

        _Bus1 = fields.get('Bus1')
        output.write(f' Bus1={_quoted(_Bus1)}')
        _kVAC = fields.get('kVAC')
        if _kVAC is not None:
            output.write(f' kVAC={_kVAC}')

        _kVDC = fields.get('kVDC')
        if _kVDC is not None:
            output.write(f' kVDC={_kVDC}')

        _kW = fields.get('kW')
        if _kW is not None:
            output.write(f' kW={_kW}')

        _NDC = fields.get('NDC')
        if _NDC is not None:
            output.write(f' NDC={_NDC}')

        _RAC = fields.get('RAC')
        if _RAC is not None:
            output.write(f' RAC={_RAC}')

        _XAC = fields.get('XAC')
        if _XAC is not None:
            output.write(f' XAC={_XAC}')

        _M0 = fields.get('M0')
        if _M0 is not None:
            output.write(f' M0={_M0}')

        _d0 = fields.get('d0')
        if _d0 is not None:
            output.write(f' d0={_d0}')

        _MMin = fields.get('MMin')
        if _MMin is not None:
            output.write(f' MMin={_MMin}')

        _MMax = fields.get('MMax')
        if _MMax is not None:
            output.write(f' MMax={_MMax}')

        _IACMax = fields.get('IACMax')
        if _IACMax is not None:
            output.write(f' IACMax={_IACMax}')

        _IDCMax = fields.get('IDCMax')
        if _IDCMax is not None:
            output.write(f' IDCMax={_IDCMax}')

        _VACRef = fields.get('VACRef')
        if _VACRef is not None:
            output.write(f' VACRef={_VACRef}')

        _PACRef = fields.get('PACRef')
        if _PACRef is not None:
            output.write(f' PACRef={_PACRef}')

        _QACRef = fields.get('QACRef')
        if _QACRef is not None:
            output.write(f' QACRef={_QACRef}')

        _VDCRef = fields.get('VDCRef')
        if _VDCRef is not None:
            output.write(f' VDCRef={_VDCRef}')

        _VSCMode = fields.get('VSCMode')
        if _VSCMode is not None:
            output.write(f' VSCMode={_quoted(_VSCMode)}')

        _Spectrum = fields.get('Spectrum')
        if _Spectrum is not None:
            output.write(f' Spectrum={_quoted(_Spectrum)}')

        _BaseFreq = fields.get('BaseFreq')
        if _BaseFreq is not None:
            output.write(f' BaseFreq={_BaseFreq}')

        _Enabled = fields.get('Enabled')
        if _Enabled is not None:
            output.write(f' Enabled={_Enabled}')

        output.write('\n')


VSConverter_ = VSConverter


class VSConverterList(RootModel[List[VSConverter]]):
    root: List[VSConverter]





class VSConverterContainer(RootModel[Union[VSConverterList, JSONFilePath, JSONLinesFilePath]]):
    root: Union[VSConverterList, JSONFilePath, JSONLinesFilePath] = Field(..., title="VSConverterContainer")


    @model_validator(mode="before")
    def _val_dss_model(self) -> "VSConverterContainer":
        try:
            _fields_set = set(self.keys())
        except:
            if isinstance(self, (list, tuple)):
                return self

            raise

        # Validate oneOf (spec. sets) based on the `required` lists
        _required_VSConverterList = "root" in _fields_set and isinstance(self.root, VSConverterList)
        _required_JSONFilePath = _fields_set.issuperset({'JSONFile'})
        _required_JSONLinesFilePath = _fields_set.issuperset({'JSONLinesFile'})
        num_specs = _required_VSConverterList + _required_JSONFilePath + _required_JSONLinesFilePath
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



class Monitor(BaseModel):
    Name: Annotated[str, Field(max_length=255, min_length=1)] = Field(..., title="Name")
    Like: Optional[str] = Field(None, title="Like")
    Element: Optional[str] = Field(None, title="Element")
    Terminal: Optional[int] = Field(None, title="Terminal")
    Mode: Optional[int] = Field(None, title="Mode")
    Residual: Optional[bool] = Field(None, title="Residual")
    VIPolar: Optional[bool] = Field(None, title="VIPolar")
    PPolar: Optional[bool] = Field(None, title="PPolar")
    BaseFreq: Optional[PositiveFloat] = Field(None, title="BaseFreq")
    Enabled: Optional[bool] = Field(None, title="Enabled")
    Action: Optional[MonitorAction] = Field(None, title="Action")

    def dumps_dss(self, edit=False) -> str:
        """
        Dump the DSS object dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            self.dump_dss(output, edit)
            return output.getvalue()

    def dump_dss(self, output: TextIO, edit=False):
        """
        Dump the DSS object to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        Monitor.dict_dump_dss(self.model_dump(exclude_unset=True), output, edit=edit)

    @staticmethod
    def dict_dumps_dss(fields: Dict, edit=False) -> str:
        """
        Dump the DSS object from a dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            Monitor.dict_dump_dss(fields, output, edit)
            return output.getvalue()

    @staticmethod
    def dict_dump_dss(fields: Dict, output: TextIO, edit=False):
        """
        Dump the DSS object from a dict to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        if edit:
            output.write(f'''edit Monitor.{fields['Name']}''')
        else:
            output.write(f'''new Monitor.{fields['Name']}''')

        _Like = fields.get('Like')
        if _Like is not None:
            output.write(f' Like={_quoted(_Like)}')

        _Element = fields.get('Element')
        if _Element is not None:
            output.write(f' Element={_quoted(_Element)}')

        _Terminal = fields.get('Terminal')
        if _Terminal is not None:
            output.write(f' Terminal={_Terminal}')

        _Mode = fields.get('Mode')
        if _Mode is not None:
            output.write(f' Mode={_Mode}')

        _Residual = fields.get('Residual')
        if _Residual is not None:
            output.write(f' Residual={_Residual}')

        _VIPolar = fields.get('VIPolar')
        if _VIPolar is not None:
            output.write(f' VIPolar={_VIPolar}')

        _PPolar = fields.get('PPolar')
        if _PPolar is not None:
            output.write(f' PPolar={_PPolar}')

        _BaseFreq = fields.get('BaseFreq')
        if _BaseFreq is not None:
            output.write(f' BaseFreq={_BaseFreq}')

        _Enabled = fields.get('Enabled')
        if _Enabled is not None:
            output.write(f' Enabled={_Enabled}')

        _Action = fields.get('Action')
        if _Action is not None:
            output.write(f' Action={_quoted(_Action)}')

        output.write('\n')


Monitor_ = Monitor


class MonitorList(RootModel[List[Monitor]]):
    root: List[Monitor]





class MonitorContainer(RootModel[Union[MonitorList, JSONFilePath, JSONLinesFilePath]]):
    root: Union[MonitorList, JSONFilePath, JSONLinesFilePath] = Field(..., title="MonitorContainer")


    @model_validator(mode="before")
    def _val_dss_model(self) -> "MonitorContainer":
        try:
            _fields_set = set(self.keys())
        except:
            if isinstance(self, (list, tuple)):
                return self

            raise

        # Validate oneOf (spec. sets) based on the `required` lists
        _required_MonitorList = "root" in _fields_set and isinstance(self.root, MonitorList)
        _required_JSONFilePath = _fields_set.issuperset({'JSONFile'})
        _required_JSONLinesFilePath = _fields_set.issuperset({'JSONLinesFile'})
        num_specs = _required_MonitorList + _required_JSONFilePath + _required_JSONLinesFilePath
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



class EnergyMeter(BaseModel):
    Name: Annotated[str, Field(max_length=255, min_length=1)] = Field(..., title="Name")
    Like: Optional[str] = Field(None, title="Like")
    Element: Optional[str] = Field(None, title="Element")
    Terminal: Optional[int] = Field(None, title="Terminal")
    Option: Optional[StringArrayOrFilePath] = Field(None, title="Option")
    kVANormal: Optional[float] = Field(None, title="kVANormal")
    kVAEmerg: Optional[float] = Field(None, title="kVAEmerg")
    PeakCurrent: Optional[List[float]] = Field(None, title="PeakCurrent")
    ZoneList: Optional[StringArrayOrFilePath] = Field(None, title="ZoneList")
    LocalOnly: Optional[bool] = Field(None, title="LocalOnly")
    Mask: Optional[List[float]] = Field(None, title="Mask")
    Losses: Optional[bool] = Field(None, title="Losses")
    LineLosses: Optional[bool] = Field(None, title="LineLosses")
    XfmrLosses: Optional[bool] = Field(None, title="XfmrLosses")
    SeqLosses: Optional[bool] = Field(None, title="SeqLosses")
    ThreePhaseLosses: Optional[bool] = Field(None, title="3PhaseLosses", validation_alias=AliasChoices("ThreePhaseLosses", "3PhaseLosses"))
    VBaseLosses: Optional[bool] = Field(None, title="VBaseLosses")
    PhaseVoltageReport: Optional[bool] = Field(None, title="PhaseVoltageReport")
    Int_Rate: Optional[float] = Field(None, title="Int_Rate")
    Int_Duration: Optional[float] = Field(None, title="Int_Duration")
    BaseFreq: Optional[PositiveFloat] = Field(None, title="BaseFreq")
    Enabled: Optional[bool] = Field(None, title="Enabled")
    Action: Optional[EnergyMeterAction] = Field(None, title="Action")

    def dumps_dss(self, edit=False) -> str:
        """
        Dump the DSS object dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            self.dump_dss(output, edit)
            return output.getvalue()

    def dump_dss(self, output: TextIO, edit=False):
        """
        Dump the DSS object to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        EnergyMeter.dict_dump_dss(self.model_dump(exclude_unset=True), output, edit=edit)

    @staticmethod
    def dict_dumps_dss(fields: Dict, edit=False) -> str:
        """
        Dump the DSS object from a dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            EnergyMeter.dict_dump_dss(fields, output, edit)
            return output.getvalue()

    @staticmethod
    def dict_dump_dss(fields: Dict, output: TextIO, edit=False):
        """
        Dump the DSS object from a dict to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        if edit:
            output.write(f'''edit EnergyMeter.{fields['Name']}''')
        else:
            output.write(f'''new EnergyMeter.{fields['Name']}''')

        _Like = fields.get('Like')
        if _Like is not None:
            output.write(f' Like={_quoted(_Like)}')

        _Element = fields.get('Element')
        if _Element is not None:
            output.write(f' Element={_quoted(_Element)}')

        _Terminal = fields.get('Terminal')
        if _Terminal is not None:
            output.write(f' Terminal={_Terminal}')

        _Option = fields.get('Option')
        if _Option is not None:
            output.write(f' Option=({_filepath_stringlist(_Option)})')

        _kVANormal = fields.get('kVANormal')
        if _kVANormal is not None:
            output.write(f' kVANormal={_kVANormal}')

        _kVAEmerg = fields.get('kVAEmerg')
        if _kVAEmerg is not None:
            output.write(f' kVAEmerg={_kVAEmerg}')

        _PeakCurrent = fields.get('PeakCurrent')
        if _PeakCurrent is not None:
            output.write(f' PeakCurrent={_as_list(_PeakCurrent)}')

        _ZoneList = fields.get('ZoneList')
        if _ZoneList is not None:
            output.write(f' ZoneList=({_filepath_stringlist(_ZoneList)})')

        _LocalOnly = fields.get('LocalOnly')
        if _LocalOnly is not None:
            output.write(f' LocalOnly={_LocalOnly}')

        _Mask = fields.get('Mask')
        if _Mask is not None:
            output.write(f' Mask={_as_list(_Mask)}')

        _Losses = fields.get('Losses')
        if _Losses is not None:
            output.write(f' Losses={_Losses}')

        _LineLosses = fields.get('LineLosses')
        if _LineLosses is not None:
            output.write(f' LineLosses={_LineLosses}')

        _XfmrLosses = fields.get('XfmrLosses')
        if _XfmrLosses is not None:
            output.write(f' XfmrLosses={_XfmrLosses}')

        _SeqLosses = fields.get('SeqLosses')
        if _SeqLosses is not None:
            output.write(f' SeqLosses={_SeqLosses}')

        _ThreePhaseLosses = fields.get('ThreePhaseLosses')
        if _ThreePhaseLosses is not None:
            output.write(f' 3PhaseLosses={_ThreePhaseLosses}')

        _VBaseLosses = fields.get('VBaseLosses')
        if _VBaseLosses is not None:
            output.write(f' VBaseLosses={_VBaseLosses}')

        _PhaseVoltageReport = fields.get('PhaseVoltageReport')
        if _PhaseVoltageReport is not None:
            output.write(f' PhaseVoltageReport={_PhaseVoltageReport}')

        _Int_Rate = fields.get('Int_Rate')
        if _Int_Rate is not None:
            output.write(f' Int_Rate={_Int_Rate}')

        _Int_Duration = fields.get('Int_Duration')
        if _Int_Duration is not None:
            output.write(f' Int_Duration={_Int_Duration}')

        _BaseFreq = fields.get('BaseFreq')
        if _BaseFreq is not None:
            output.write(f' BaseFreq={_BaseFreq}')

        _Enabled = fields.get('Enabled')
        if _Enabled is not None:
            output.write(f' Enabled={_Enabled}')

        _Action = fields.get('Action')
        if _Action is not None:
            output.write(f' Action={_quoted(_Action)}')

        output.write('\n')


EnergyMeter_ = EnergyMeter


class EnergyMeterList(RootModel[List[EnergyMeter]]):
    root: List[EnergyMeter]





class EnergyMeterContainer(RootModel[Union[EnergyMeterList, JSONFilePath, JSONLinesFilePath]]):
    root: Union[EnergyMeterList, JSONFilePath, JSONLinesFilePath] = Field(..., title="EnergyMeterContainer")


    @model_validator(mode="before")
    def _val_dss_model(self) -> "EnergyMeterContainer":
        try:
            _fields_set = set(self.keys())
        except:
            if isinstance(self, (list, tuple)):
                return self

            raise

        # Validate oneOf (spec. sets) based on the `required` lists
        _required_EnergyMeterList = "root" in _fields_set and isinstance(self.root, EnergyMeterList)
        _required_JSONFilePath = _fields_set.issuperset({'JSONFile'})
        _required_JSONLinesFilePath = _fields_set.issuperset({'JSONLinesFile'})
        num_specs = _required_EnergyMeterList + _required_JSONFilePath + _required_JSONLinesFilePath
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



class Sensor_Common(BaseModel):
    Name: Annotated[str, Field(max_length=255, min_length=1)] = Field(..., title="Name")
    Like: Optional[str] = Field(None, title="Like")
    Element: str = Field(..., title="Element")
    Terminal: Optional[int] = Field(None, title="Terminal")
    kVBase: float = Field(..., title="kVBase")
    kVs: Optional[List[float]] = Field(None, title="kVs")
    Conn: Optional[Connection] = Field(None, title="Conn")
    DeltaDirection: Optional[int] = Field(None, title="DeltaDirection")
    pctError: Optional[float] = Field(None, title="%Error", validation_alias=AliasChoices("pctError", "%Error"))
    Weight: Optional[float] = Field(None, title="Weight")
    BaseFreq: Optional[PositiveFloat] = Field(None, title="BaseFreq")
    Enabled: Optional[bool] = Field(None, title="Enabled")
    Clear: Optional[bool] = Field(None, title="Clear")

class Sensor_kWskvars(Sensor_Common):
    kWs: List[float] = Field(..., title="kWs")
    kvars: Optional[List[float]] = Field(None, title="kvars")

class Sensor_currents(Sensor_Common):
    Currents: List[float] = Field(..., title="Currents")


class Sensor(RootModel[Union[Sensor_kWskvars, Sensor_currents]]):
    root: Union[Sensor_kWskvars, Sensor_currents] = Field(..., title="Sensor")

    def dumps_dss(self, edit=False) -> str:
        """
        Dump the DSS object dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            self.dump_dss(output, edit)
            return output.getvalue()

    def dump_dss(self, output: TextIO, edit=False):
        """
        Dump the DSS object to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        Sensor.dict_dump_dss(self.model_dump(exclude_unset=True), output, edit=edit)

    @staticmethod
    def dict_dumps_dss(fields: Dict, edit=False) -> str:
        """
        Dump the DSS object from a dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            Sensor.dict_dump_dss(fields, output, edit)
            return output.getvalue()

    @staticmethod
    def dict_dump_dss(fields: Dict, output: TextIO, edit=False):
        """
        Dump the DSS object from a dict to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        if edit:
            output.write(f'''edit Sensor.{fields['Name']}''')
        else:
            output.write(f'''new Sensor.{fields['Name']}''')

        _Like = fields.get('Like')
        if _Like is not None:
            output.write(f' Like={_quoted(_Like)}')

        _Element = fields.get('Element')
        output.write(f' Element={_quoted(_Element)}')
        _Terminal = fields.get('Terminal')
        if _Terminal is not None:
            output.write(f' Terminal={_Terminal}')

        _kVBase = fields.get('kVBase')
        output.write(f' kVBase={_kVBase}')
        _kVs = fields.get('kVs')
        if _kVs is not None:
            output.write(f' kVs={_as_list(_kVs)}')

        _Currents = fields.get('Currents')
        if _Currents is not None:
            output.write(f' Currents={_as_list(_Currents)}')

        _kWs = fields.get('kWs')
        if _kWs is not None:
            output.write(f' kWs={_as_list(_kWs)}')

        _kvars = fields.get('kvars')
        if _kvars is not None:
            output.write(f' kvars={_as_list(_kvars)}')

        _Conn = fields.get('Conn')
        if _Conn is not None:
            output.write(f' Conn={_quoted(_Conn)}')

        _DeltaDirection = fields.get('DeltaDirection')
        if _DeltaDirection is not None:
            output.write(f' DeltaDirection={_DeltaDirection}')

        _pctError = fields.get('pctError')
        if _pctError is not None:
            output.write(f' %Error={_pctError}')

        _Weight = fields.get('Weight')
        if _Weight is not None:
            output.write(f' Weight={_Weight}')

        _BaseFreq = fields.get('BaseFreq')
        if _BaseFreq is not None:
            output.write(f' BaseFreq={_BaseFreq}')

        _Enabled = fields.get('Enabled')
        if _Enabled is not None:
            output.write(f' Enabled={_Enabled}')

        _Clear = fields.get('Clear')
        if _Clear is not None:
            output.write(f' Clear={_Clear}')

        output.write('\n')

    @model_validator(mode="before")
    def _val_dss_model(self) -> "Sensor":
        _fields_set = set(self.keys())
        # Validate oneOf (spec. sets) based on the `required` lists
        _required_kWskvars = _fields_set.issuperset({'kWs'})
        _required_currents = _fields_set.issuperset({'Currents'})
        num_specs = _required_kWskvars + _required_currents
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



Sensor_ = Sensor


class SensorList(RootModel[List[Sensor]]):
    root: List[Sensor]





class SensorContainer(RootModel[Union[SensorList, JSONFilePath, JSONLinesFilePath]]):
    root: Union[SensorList, JSONFilePath, JSONLinesFilePath] = Field(..., title="SensorContainer")


    @model_validator(mode="before")
    def _val_dss_model(self) -> "SensorContainer":
        try:
            _fields_set = set(self.keys())
        except:
            if isinstance(self, (list, tuple)):
                return self

            raise

        # Validate oneOf (spec. sets) based on the `required` lists
        _required_SensorList = "root" in _fields_set and isinstance(self.root, SensorList)
        _required_JSONFilePath = _fields_set.issuperset({'JSONFile'})
        _required_JSONLinesFilePath = _fields_set.issuperset({'JSONLinesFile'})
        num_specs = _required_SensorList + _required_JSONFilePath + _required_JSONLinesFilePath
        if num_specs > 1:
            raise ValueError("AltDSS: Conflict detected in the provided properties. Only one specification type is allowed.")
        elif num_specs == 0:
            raise ValueError("AltDSS: Model is not fully specified; no submodel is fully satisfied.")

        return self



class Circuit(BaseModel):
    Name: Optional[Annotated[str, Field(max_length=255, min_length=1)]] = Field(None, title="Name")
    DefaultBaseFreq: Optional[PositiveFloat] = Field(None, title="DefaultBaseFreq")
    PreCommands: Optional[List[str]] = Field(None, title="PreCommands")
    PostCommands: Optional[List[str]] = Field(None, title="PostCommands")
    Bus: Optional[List[Bus_]] = Field(None, title="Bus")
    LineCode: Optional[LineCodeContainer] = Field(None, title="LineCode")
    LoadShape: Optional[LoadShapeContainer] = Field(None, title="LoadShape")
    TShape: Optional[TShapeContainer] = Field(None, title="TShape")
    PriceShape: Optional[PriceShapeContainer] = Field(None, title="PriceShape")
    XYcurve: Optional[XYcurveContainer] = Field(None, title="XYcurve")
    GrowthShape: Optional[GrowthShapeContainer] = Field(None, title="GrowthShape")
    TCC_Curve: Optional[TCC_CurveContainer] = Field(None, title="TCC_Curve")
    Spectrum: Optional[SpectrumContainer] = Field(None, title="Spectrum")
    WireData: Optional[WireDataContainer] = Field(None, title="WireData")
    CNData: Optional[CNDataContainer] = Field(None, title="CNData")
    TSData: Optional[TSDataContainer] = Field(None, title="TSData")
    LineSpacing: Optional[LineSpacingContainer] = Field(None, title="LineSpacing")
    LineGeometry: Optional[LineGeometryContainer] = Field(None, title="LineGeometry")
    XfmrCode: Optional[XfmrCodeContainer] = Field(None, title="XfmrCode")
    Line: Optional[LineContainer] = Field(None, title="Line")
    Vsource: VsourceContainer = Field(..., title="Vsource")
    Isource: Optional[IsourceContainer] = Field(None, title="Isource")
    VCCS: Optional[VCCSContainer] = Field(None, title="VCCS")
    Load: Optional[LoadContainer] = Field(None, title="Load")
    Transformer: Optional[TransformerContainer] = Field(None, title="Transformer")
    RegControl: Optional[RegControlContainer] = Field(None, title="RegControl")
    Capacitor: Optional[CapacitorContainer] = Field(None, title="Capacitor")
    Reactor: Optional[ReactorContainer] = Field(None, title="Reactor")
    CapControl: Optional[CapControlContainer] = Field(None, title="CapControl")
    Fault: Optional[FaultContainer] = Field(None, title="Fault")
    DynamicExp: Optional[DynamicExpContainer] = Field(None, title="DynamicExp")
    Generator: Optional[GeneratorContainer] = Field(None, title="Generator")
    GenDispatcher: Optional[GenDispatcherContainer] = Field(None, title="GenDispatcher")
    Storage: Optional[StorageContainer] = Field(None, title="Storage")
    StorageController: Optional[StorageControllerContainer] = Field(None, title="StorageController")
    Relay: Optional[RelayContainer] = Field(None, title="Relay")
    Recloser: Optional[RecloserContainer] = Field(None, title="Recloser")
    Fuse: Optional[FuseContainer] = Field(None, title="Fuse")
    SwtControl: Optional[SwtControlContainer] = Field(None, title="SwtControl")
    PVSystem: Optional[PVSystemContainer] = Field(None, title="PVSystem")
    UPFC: Optional[UPFCContainer] = Field(None, title="UPFC")
    UPFCControl: Optional[UPFCControlContainer] = Field(None, title="UPFCControl")
    ESPVLControl: Optional[ESPVLControlContainer] = Field(None, title="ESPVLControl")
    IndMach012: Optional[IndMach012Container] = Field(None, title="IndMach012")
    GICsource: Optional[GICsourceContainer] = Field(None, title="GICsource")
    AutoTrans: Optional[AutoTransContainer] = Field(None, title="AutoTrans")
    InvControl: Optional[InvControlContainer] = Field(None, title="InvControl")
    ExpControl: Optional[ExpControlContainer] = Field(None, title="ExpControl")
    GICLine: Optional[GICLineContainer] = Field(None, title="GICLine")
    GICTransformer: Optional[GICTransformerContainer] = Field(None, title="GICTransformer")
    VSConverter: Optional[VSConverterContainer] = Field(None, title="VSConverter")
    Monitor: Optional[MonitorContainer] = Field(None, title="Monitor")
    EnergyMeter: Optional[EnergyMeterContainer] = Field(None, title="EnergyMeter")
    Sensor: Optional[SensorContainer] = Field(None, title="Sensor")

    def dumps_dss(self, edit=False) -> str:
        """
        Dump the DSS object dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            self.dump_dss(output, edit)
            return output.getvalue()

    def dump_dss(self, output: TextIO, edit=False):
        """
        Dump the DSS object to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        Circuit.dict_dump_dss(self.model_dump(exclude_unset=True), output, edit=edit)

    @staticmethod
    def dict_dumps_dss(fields: Dict, edit=False) -> str:
        """
        Dump the DSS object from a dict to a string.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.

        Convenience function.
        """
        with StringIO() as output:
            Circuit.dict_dump_dss(fields, output, edit)
            return output.getvalue()

    @staticmethod
    def dict_dump_dss(fields: Dict, output: TextIO, edit=False):
        """
        Dump the DSS object from a dict to the `output` stream.
        Use `edit=True` to emit an edit DSS command instead of the
        default `new` command.
        """
        
        output.write('clear\n')
        _DefaultBaseFreq = fields.get('DefaultBaseFreq')
        if _DefaultBaseFreq is not None:
            output.write(f'set DefaultBaseFreq={_DefaultBaseFreq}\n')
        
        output.write(f'''new Circuit.{fields['Name']}\n''')
        
        for command in fields.get('PreCommands', []):
            output.write(command)
            output.write('\n')
        
        
        _LineCode = fields.get('LineCode')
        if _LineCode is not None:
            _dump_dss_container(_LineCode, LineCode, output)

        _LoadShape = fields.get('LoadShape')
        if _LoadShape is not None:
            _dump_dss_container(_LoadShape, LoadShape, output)

        _TShape = fields.get('TShape')
        if _TShape is not None:
            _dump_dss_container(_TShape, TShape, output)

        _PriceShape = fields.get('PriceShape')
        if _PriceShape is not None:
            _dump_dss_container(_PriceShape, PriceShape, output)

        _XYcurve = fields.get('XYcurve')
        if _XYcurve is not None:
            _dump_dss_container(_XYcurve, XYcurve, output)

        _GrowthShape = fields.get('GrowthShape')
        if _GrowthShape is not None:
            _dump_dss_container(_GrowthShape, GrowthShape, output)

        _TCC_Curve = fields.get('TCC_Curve')
        if _TCC_Curve is not None:
            _dump_dss_container(_TCC_Curve, TCC_Curve, output)

        _Spectrum = fields.get('Spectrum')
        if _Spectrum is not None:
            _dump_dss_container(_Spectrum, Spectrum, output)

        _WireData = fields.get('WireData')
        if _WireData is not None:
            _dump_dss_container(_WireData, WireData, output)

        _CNData = fields.get('CNData')
        if _CNData is not None:
            _dump_dss_container(_CNData, CNData, output)

        _TSData = fields.get('TSData')
        if _TSData is not None:
            _dump_dss_container(_TSData, TSData, output)

        _LineSpacing = fields.get('LineSpacing')
        if _LineSpacing is not None:
            _dump_dss_container(_LineSpacing, LineSpacing, output)

        _LineGeometry = fields.get('LineGeometry')
        if _LineGeometry is not None:
            _dump_dss_container(_LineGeometry, LineGeometry, output)

        _XfmrCode = fields.get('XfmrCode')
        if _XfmrCode is not None:
            _dump_dss_container(_XfmrCode, XfmrCode, output)

        _Line = fields.get('Line')
        if _Line is not None:
            _dump_dss_container(_Line, Line, output)

        _Vsource = fields.get('Vsource')
        _dump_dss_container(_Vsource, Vsource, output)
        _Isource = fields.get('Isource')
        if _Isource is not None:
            _dump_dss_container(_Isource, Isource, output)

        _VCCS = fields.get('VCCS')
        if _VCCS is not None:
            _dump_dss_container(_VCCS, VCCS, output)

        _Load = fields.get('Load')
        if _Load is not None:
            _dump_dss_container(_Load, Load, output)

        _Transformer = fields.get('Transformer')
        if _Transformer is not None:
            _dump_dss_container(_Transformer, Transformer, output)

        _RegControl = fields.get('RegControl')
        if _RegControl is not None:
            _dump_dss_container(_RegControl, RegControl, output)

        _Capacitor = fields.get('Capacitor')
        if _Capacitor is not None:
            _dump_dss_container(_Capacitor, Capacitor, output)

        _Reactor = fields.get('Reactor')
        if _Reactor is not None:
            _dump_dss_container(_Reactor, Reactor, output)

        _CapControl = fields.get('CapControl')
        if _CapControl is not None:
            _dump_dss_container(_CapControl, CapControl, output)

        _Fault = fields.get('Fault')
        if _Fault is not None:
            _dump_dss_container(_Fault, Fault, output)

        _DynamicExp = fields.get('DynamicExp')
        if _DynamicExp is not None:
            _dump_dss_container(_DynamicExp, DynamicExp, output)

        _Generator = fields.get('Generator')
        if _Generator is not None:
            _dump_dss_container(_Generator, Generator, output)

        _GenDispatcher = fields.get('GenDispatcher')
        if _GenDispatcher is not None:
            _dump_dss_container(_GenDispatcher, GenDispatcher, output)

        _Storage = fields.get('Storage')
        if _Storage is not None:
            _dump_dss_container(_Storage, Storage, output)

        _StorageController = fields.get('StorageController')
        if _StorageController is not None:
            _dump_dss_container(_StorageController, StorageController, output)

        _Relay = fields.get('Relay')
        if _Relay is not None:
            _dump_dss_container(_Relay, Relay, output)

        _Recloser = fields.get('Recloser')
        if _Recloser is not None:
            _dump_dss_container(_Recloser, Recloser, output)

        _Fuse = fields.get('Fuse')
        if _Fuse is not None:
            _dump_dss_container(_Fuse, Fuse, output)

        _SwtControl = fields.get('SwtControl')
        if _SwtControl is not None:
            _dump_dss_container(_SwtControl, SwtControl, output)

        _PVSystem = fields.get('PVSystem')
        if _PVSystem is not None:
            _dump_dss_container(_PVSystem, PVSystem, output)

        _UPFC = fields.get('UPFC')
        if _UPFC is not None:
            _dump_dss_container(_UPFC, UPFC, output)

        _UPFCControl = fields.get('UPFCControl')
        if _UPFCControl is not None:
            _dump_dss_container(_UPFCControl, UPFCControl, output)

        _ESPVLControl = fields.get('ESPVLControl')
        if _ESPVLControl is not None:
            _dump_dss_container(_ESPVLControl, ESPVLControl, output)

        _IndMach012 = fields.get('IndMach012')
        if _IndMach012 is not None:
            _dump_dss_container(_IndMach012, IndMach012, output)

        _GICsource = fields.get('GICsource')
        if _GICsource is not None:
            _dump_dss_container(_GICsource, GICsource, output)

        _AutoTrans = fields.get('AutoTrans')
        if _AutoTrans is not None:
            _dump_dss_container(_AutoTrans, AutoTrans, output)

        _InvControl = fields.get('InvControl')
        if _InvControl is not None:
            _dump_dss_container(_InvControl, InvControl, output)

        _ExpControl = fields.get('ExpControl')
        if _ExpControl is not None:
            _dump_dss_container(_ExpControl, ExpControl, output)

        _GICLine = fields.get('GICLine')
        if _GICLine is not None:
            _dump_dss_container(_GICLine, GICLine, output)

        _GICTransformer = fields.get('GICTransformer')
        if _GICTransformer is not None:
            _dump_dss_container(_GICTransformer, GICTransformer, output)

        _VSConverter = fields.get('VSConverter')
        if _VSConverter is not None:
            _dump_dss_container(_VSConverter, VSConverter, output)

        _Monitor = fields.get('Monitor')
        if _Monitor is not None:
            _dump_dss_container(_Monitor, Monitor, output)

        _EnergyMeter = fields.get('EnergyMeter')
        if _EnergyMeter is not None:
            _dump_dss_container(_EnergyMeter, EnergyMeter, output)

        _Sensor = fields.get('Sensor')
        if _Sensor is not None:
            _dump_dss_container(_Sensor, Sensor, output)

        output.write('MakeBusList\n')
        for bus in fields.get('Bus', []):
            Bus.dict_dump_dss(bus, output)
        
        for command in fields.get('PostCommands', []):
            output.write(command)
            output.write('\n')
        
        output.write('\n')
        


