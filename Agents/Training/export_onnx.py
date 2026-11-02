import argparse
import copy
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import lightning as L
from state_value_net import StateValueModule


class ExportedValueNet(nn.Module):
    """Outputs für den Export: die Value Logits, bei Modellen mit trainiertem Platzierungs-Head die Platzierungs-Logits
    [batch, 4 Spieler, 4 Plätze], bei Modellen mit Policy Head die Queries [batch, Typen * dim] und die Embedding-Tabelle
    der Zug-Tokens [Vokabular, dim] (konstant, für alle Batches gleich). Die Queries kommen als float16: sie sind 300-mal so
    groß wie die Value Logits, und die Kopie von der GPU bremst sonst die Suche"""

    def __init__(self, model: nn.Module, with_placements: bool, with_policy: bool, fp16: bool = False):
        super().__init__()
        self.with_placements, self.with_policy, self.fp16 = with_placements, with_policy, fp16
        self.input_mean = self.input_scale = None
        if fp16:
            # Die Schichten in float16 (Tensor Cores, halber Speicherverkehr); Input und Outputs bleiben wie beim
            # float32-Export, die Standardisierung der rohen Features rechnet vorher in float32
            model = copy.deepcopy(model)
            if model.input_mean is not None:
                self.input_mean, self.input_scale = model.input_mean.float(), model.input_scale.float()
                model.input_mean = model.input_scale = None
            model = model.half()
        self.model = model

    def forward(self, x):
        raw = x
        if self.fp16:
            if self.input_mean is not None:
                x = (x - self.input_mean) * self.input_scale
            x = x.half()
        # Block-Encoder lesen die Koordinaten des Räubers aus den unskalierten Features
        outputs = self.model.forward_all(x, raw)
        result = [outputs["logits"].float()]
        if self.with_placements:
            result.append(outputs["placements"].float())
        if self.with_policy:
            result.append(outputs["policy_query"].flatten(1).half())
            result.append(self.model.policy_embeddings.weight.float() * 1.0)
        return tuple(result)


def export(ckpt_path: str, out_path: str, in_dim: int = None, fp16: bool = False):
    # Lightning-Module aus Checkpoint laden, die Inputgröße steht in seinen Hyperparametern
    overrides = {"in_dim": in_dim} if in_dim else {}
    module = StateValueModule.load_from_checkpoint(ckpt_path, map_location="cpu", weights_only=True, **overrides)
    in_dim = module.hparams.in_dim

    # Dropout deaktivieren und freezen
    module.eval()

    # Kern-Modell statt Lightning-Module für ONNX-Export nötig
    export_model(module.model, out_path, in_dim, fp16)


def export_model(model: nn.Module, out_path: str, in_dim: int, fp16: bool = False) -> float:
    """Exportiert ein StateValueNet (im eval-Modus) nach ONNX und prüft die Parität.
    Modelle mit trainiertem Platzierungs-Head (--placement-weight > 0) bekommen einen zweiten Output "placements"
    (der C#-Code liest "logits" über den Namen); Aux Heads nur für andere Losses werden nicht exportiert.
    Mit fp16 rechnen die Schichten in float16 (Input und Outputs wie bei float32), die Parität dann gegen das
    float32-Modell mit der Toleranz von float16.
    Gibt die maximale Abweichung zwischen PyTorch und ONNX zurück."""
    with_placements = getattr(model, "aux", None) is not None and getattr(model, "placements_trained", False)
    with_policy = getattr(model, "policy_query", None) is not None
    exported = ExportedValueNet(model, with_placements, with_policy, fp16).eval()
    reference = ExportedValueNet(model, with_placements, with_policy).eval()
    output_names = ["logits"] + (["placements"] if with_placements else []) + (["policy_query", "policy_embeddings"] if with_policy else [])
    batch_outputs = [name for name in output_names if name != "policy_embeddings"]

    dummy = torch.randn(1, in_dim)
    torch.onnx.export(
        exported, dummy, out_path,
        input_names=["state"], output_names=output_names,
        dynamic_axes={"state": {0: "batch"}, **{name: {0: "batch"} for name in batch_outputs}},
        opset_version=17, dynamo=False,
    )
    if with_policy:
        _name_constant_outputs(out_path, output_names)
    print(f"Exported to {out_path} (outputs: {', '.join(output_names)})")

    # Paritätscheck zwischen Torch und ONNX auf Random Inputs, für jeden Output
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(out_path, sess_options=so, providers=["CPUExecutionProvider"])

    def max_difference(batch: int) -> float:
        x_np = np.random.randn(batch, in_dim).astype(np.float32)
        with torch.no_grad():
            torch_out = reference(torch.from_numpy(x_np))
        torch_outputs = [t.numpy() for t in torch_out]
        onnx_outputs = sess.run(output_names, {"state": x_np})

        def difference(t, o):
            # float16-Outputs relativ zu ihrer Größe (ein paar Rundungsschritte von float16 sind erlaubt)
            if t.dtype == np.float16:
                t, o = t.astype(np.float32), o.astype(np.float32)
                return float((np.abs(t - o) / np.maximum(np.abs(t), 1.0)).max()) * 1e-2
            # Relativ zum größten Output: zufällige Inputs treiben manche Netze auf Logits in den Hunderten, wo float32
            # schon in der vierten Nachkommastelle rundet
            return float(np.abs(t - o).max() / max(1.0, float(np.abs(t).max())))
        return max(difference(t, o) for t, o in zip(torch_outputs, onnx_outputs))

    max_diff = max_difference(16)
    print(f"Max diff (relative to the largest output), PyTorch vs ONNX (batch=16): {max_diff:.2e}")
    # float16 hat ~3 Dezimalstellen, über vier Schichten summieren sich ein paar Rundungsschritte
    tolerance = 5e-2 if fp16 else 1e-4
    assert max_diff < tolerance, "ONNX export does not match PyTorch output -- do not ship this .onnx file"
    print("Parity check passed.")

    # Und nochmal mit batch=1 wegen Relevanz für MCTS
    max_diff1 = max_difference(1)
    print(f"Max diff, batch=1: {max_diff1:.2e}")

    return float(max(max_diff, max_diff1))


def _name_constant_outputs(path: str, output_names):
    """Der Export faltet die konstante Embedding-Tabelle zu einem Initializer und verliert dabei den Output-Namen: die
    Outputs der Reihe nach über Identity-Knoten wieder benennen"""
    import onnx
    model = onnx.load(path)
    for output, name in zip(model.graph.output, output_names):
        if output.name != name:
            model.graph.node.append(onnx.helper.make_node("Identity", [output.name], [name]))
            output.name = name
    onnx.save(model, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default="state_value_net.onnx")
    # Nur für Checkpoints ohne gespeicherte Inputgröße nötig
    parser.add_argument("--in-dim", type=int, default=None)
    parser.add_argument("--fp16", action="store_true", help="Schichten in float16 (Input und Outputs bleiben wie bei float32)")
    args = parser.parse_args()
    export(args.ckpt, args.out, args.in_dim, args.fp16)
