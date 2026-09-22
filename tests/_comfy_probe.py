import comfy.ops, comfy.model_patcher, comfy.sd
import comfy.ldm.modules.attention as attn_mod
import comfyui_superl8
from comfyui_superl8.ops import FNI8Ops
from comfyui_superl8._comfy_compat import COMFY

print("COMFY =", COMFY)
print("nodes =", list(comfyui_superl8.NODE_CLASS_MAPPINGS))
print("FNI8Ops <: manual_cast        :", issubclass(FNI8Ops, comfy.ops.manual_cast))
print("FNI8Ops.Linear <: mc.Linear   :", issubclass(FNI8Ops.Linear, comfy.ops.manual_cast.Linear))
print("Linear overrides fccw         :", "forward_comfy_cast_weights" in FNI8Ops.Linear.__dict__)
print("ModelPatcher.set_attn1_replace:", hasattr(comfy.model_patcher.ModelPatcher, "set_model_attn1_replace"))
print("ModelPatcher.add_object_patch :", hasattr(comfy.model_patcher.ModelPatcher, "add_object_patch"))
print("sd.load_diffusion_model_sd    :", hasattr(comfy.sd, "load_diffusion_model_state_dict"))
print("attention.optimized_attention :", hasattr(attn_mod, "optimized_attention"))
for name, cls in comfyui_superl8.NODE_CLASS_MAPPINGS.items():
    assert hasattr(cls, "INPUT_TYPES") and cls.RETURN_TYPES and cls.FUNCTION, name
print("node contract OK for all nodes")
# instantiate a FNI8Ops.Linear on meta and confirm it builds like a comfy Linear
m = FNI8Ops.Linear(8, 16, bias=False, device="meta")
print("FNI8Ops.Linear builds         :", type(m).__mro__[1].__name__)
