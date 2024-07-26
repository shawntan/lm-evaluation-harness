from dolomite_engine.hf_models import export_to_huggingface
import sys
export_to_huggingface(
    pretrained_model_name_or_path=sys.argv[1],
    save_path="hf_compatible_model",
    model_type="llama",
)
