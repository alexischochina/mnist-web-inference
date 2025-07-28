import torch
from model import MNISTNet

def export_to_onnx():
    # Charger le modèle
    model = MNISTNet()
    model.load_state_dict(torch.load('models/mnist_best.pth', map_location='cpu'))
    model.eval()

    # Créer un exemple d'entrée
    dummy_input = torch.randn(1, 1, 28, 28)

    # Exporter en ONNX
    torch.onnx.export(
        model,
        dummy_input,
        "models/mnist.onnx",
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print("Modèle exporté avec succès vers models/mnist.onnx")

if __name__ == '__main__':
    export_to_onnx() 