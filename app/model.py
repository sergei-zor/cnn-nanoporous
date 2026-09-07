import torch
from generate_model import generate_model


def load_inference_model(model_path: str, model_name: str, device: torch.device):

    model = generate_model(
        model_name=model_name,
        use_descriptors=False,
        fc_first_size=0,
        fc_second_size=0,
        p_dropout=0,
    )

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    fully_connected = model.fc
    model.fc = torch.nn.Identity()

    model.to(device)
    fully_connected.to(device)
    model.eval() 

    return model, fully_connected