# Based on https://docs.pydantic.dev/latest/concepts/types/#handling-third-party-types
from typing import Any, List
from pydantic_core import core_schema
from typing_extensions import Annotated

from pydantic import (
    BaseModel,
    Field,
    GetCoreSchemaHandler,
    GetJsonSchemaHandler,
    ValidationError,
)
from pydantic.json_schema import JsonSchemaValue

class _ComplexPydanticAnnotation:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        """
        * lists of floats of length 2 will be parsed as `complex`
        * `complex` instances will be parsed as `complex` instances without any changes
        * Nothing else will pass validation
        * Serialization will always return just a list of 2 elements
        """
        def validate_from_list(value: Annotated[List, Field(min_length=2, max_length=2)]) -> complex:
            return complex(*value)

        from_list_schema = core_schema.chain_schema(
            [
                core_schema.list_schema(items_schema=core_schema.float_schema(), min_length=2, max_length=2),
                core_schema.no_info_plain_validator_function(validate_from_list),
            ]
        )
        return core_schema.json_or_python_schema(
            json_schema=from_list_schema,
            python_schema=core_schema.union_schema([core_schema.is_instance_schema(complex), from_list_schema]),
            serialization=core_schema.plain_serializer_function_ser_schema(lambda z: [z.real, z.imag] if isinstance(z, complex) else z),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, _core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        return handler(core_schema.list_schema(items_schema=core_schema.float_schema(), min_length=2, max_length=2))

Complex = Annotated[complex, _ComplexPydanticAnnotation]
__all__ = ['Complex']
