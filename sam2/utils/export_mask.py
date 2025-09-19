import torch
import onnx
from onnx import checker

from sam2.modeling.sam.mask_decoder import MaskDecoder
from sam2.modeling.sam.transformer import TwoWayTransformer


class MaskDecoderExportWrapper(torch.nn.Module):
    def __init__(self, mask_decoder: MaskDecoder, multimask_output: bool = True, repeat_image: bool = False):
        super().__init__()
        self.mask_decoder = mask_decoder
        self.multimask_output = multimask_output
        self.repeat_image = repeat_image

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        high_res_feat0: torch.Tensor = None,
        high_res_feat1: torch.Tensor = None,
    ):
        high_res_features = None
        if getattr(self.mask_decoder, "use_high_res_features", False):
            # Si el modelo espera high-res features, deben venir ambas
            if (high_res_feat0 is None) or (high_res_feat1 is None):
                raise RuntimeError(
                    "MaskDecoder instanced on use_high_res_features=True, "
                    "load high_res_feat0 and high_res_feat1."
                )
            high_res_features = [high_res_feat0, high_res_feat1]

        masks, iou_pred, sam_tokens_out, object_score_logits = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=self.multimask_output,
            repeat_image=self.repeat_image,
            high_res_features=high_res_features,
        )
        return masks, iou_pred, sam_tokens_out, object_score_logits


def export_mask_decoder_onnx_from_checkpoint(
    ckpt_path="../../../checkpoints/edgetam.pt",
    onnx_path="mask_decoder.onnx",
    batch_size=1,
    embedding_dim=256,   
    h=64,
    w=64,
    num_heads=8,         
    depth=2,             
    mlp_dim=2048,        
    use_high_res_features=True,
    multimask_output=True,
    repeat_image=False,
):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model"]

    transformer = TwoWayTransformer(
        depth=depth,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        mlp_dim=mlp_dim,  
    )

    mask_decoder = MaskDecoder(
        transformer_dim=embedding_dim,
        transformer=transformer,
        num_multimask_outputs=3,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        use_high_res_features=use_high_res_features,   
        pred_obj_scores=True,                          
        pred_obj_scores_mlp=True,                      
    )

    md_sd = {k.replace("sam_mask_decoder.", ""): v
             for k, v in state_dict.items() if k.startswith("sam_mask_decoder.")}
    missing, unexpected = mask_decoder.load_state_dict(md_sd, strict=True)
    mask_decoder.eval()

    export_model = MaskDecoderExportWrapper(
        mask_decoder,
        multimask_output=multimask_output,
        repeat_image=repeat_image,
    )
    export_model.eval()

    # image_embeddings: [B, C, H, W]
    image_embeddings = torch.randn(batch_size, embedding_dim, h, w)
    # image_pe: [1, C, H, W] 
    image_pe = torch.randn(1, embedding_dim, h, w)
    # sparse_prompt_embeddings: [B, N_tokens_prompt, C]
    sparse_prompt_embeddings = torch.randn(batch_size, 6, embedding_dim)
    # dense_prompt_embeddings: [B, C, H, W]
    dense_prompt_embeddings = torch.randn(batch_size, embedding_dim, h, w)

    if use_high_res_features:
        high_res_feat1 = torch.randn(batch_size, embedding_dim // 4, h * 2, w * 2)
        high_res_feat0 = torch.randn(batch_size, embedding_dim // 8, h * 4, w * 4)
    else:
        high_res_feat0 = None
        high_res_feat1 = None

    with torch.no_grad():
        torch.onnx.export(
            export_model,
            (
                image_embeddings,
                image_pe,
                sparse_prompt_embeddings,
                dense_prompt_embeddings,
                high_res_feat0,
                high_res_feat1,
            ),
            onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=False,   
            input_names=[
                "image_embeddings",
                "image_pe",
                "sparse_prompt_embeddings",
                "dense_prompt_embeddings",
                "high_res_feat0",
                "high_res_feat1",
            ],
            output_names=[
                "masks",
                "iou_pred",
                "sam_tokens_out",
                "object_score_logits",
            ],
            dynamic_axes=None,
        )

    model_onnx = onnx.load(onnx_path)
    checker.check_model(model_onnx)

if __name__ == "__main__":
    export_mask_decoder_onnx_from_checkpoint()
