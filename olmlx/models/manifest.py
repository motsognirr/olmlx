import dataclasses
import json
import hashlib
import typing
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class ModelManifest:
    name: str
    hf_path: str
    size: int = 0
    modified_at: str = ""
    digest: str = ""
    format: str = "mlx"
    family: str = ""
    parameter_size: str = ""
    quantization_level: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ModelManifest":
        with open(path) as f:
            data = json.load(f)
        # Coerce None/missing to field defaults; raise on null/missing required fields
        field_names = set()
        for field in dataclasses.fields(cls):
            k = field.name
            field_names.add(k)
            is_required = (
                field.default is dataclasses.MISSING
                and field.default_factory is dataclasses.MISSING
            )
            if k not in data or data[k] is None:
                if is_required:
                    raise ValueError(
                        f"Required field '{k}' is null or missing in manifest {path}"
                    )
                data[k] = (
                    field.default
                    if field.default is not dataclasses.MISSING
                    else field.default_factory()
                )
        # Validate types for non-null values
        hints = typing.get_type_hints(cls)
        for field in dataclasses.fields(cls):
            k = field.name
            if k in data and data[k] is not None:
                field_type = hints[k]
                if (
                    field_type in (str, int)
                    and not isinstance(data[k], bool)
                    and not isinstance(data[k], field_type)
                ):
                    raise ValueError(
                        f"Field '{k}' should be {field_type.__name__}, "
                        f"got {type(data[k]).__name__} in {path}"
                    )
        return cls(**{k: v for k, v in data.items() if k in field_names})

    @staticmethod
    def compute_digest(name: str) -> str:
        return "sha256:" + hashlib.sha256(name.encode()).hexdigest()[:12]
