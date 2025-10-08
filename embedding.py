import sys
from typing import Optional

from embedding import EmbeddingPipeline, load_config


def main(input_glob: Optional[str] = None) -> None:
    config = load_config()
    pipeline = EmbeddingPipeline(config)
    pattern = input_glob or "test/*.txt"
    pipeline.run(pattern)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(arg)
