import dataclasses
import json
import hashlib
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
        for k, field in cls.__dataclass_fields__.items():
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
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @staticmethod
    def compute_digest(name: str) -> str:
        return "sha256:" + hashlib.sha256(name.encode()).hexdigest()[:12]
