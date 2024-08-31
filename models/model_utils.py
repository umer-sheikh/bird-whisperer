import torch
import torchvision
from .whisper.whisper_model import WhisperModel


def get_model(model_name='whisper', pre_trained=True, n_classes=264):

	assert n_classes is not None, "'n_classes' cannot be None. Specify 'n_classes' present in the dataset."
	
	print(f"\n########################\nModel Information\n########################\n")
	print(f"Model Name: {model_name}\nPre-Trained: {pre_trained}")

	if model_name == 'efficientnet_b4':
		if pre_trained:
			model = torchvision.models.efficientnet_b4(weights='IMAGENET1K_V1')
		else:
			model = torchvision.models.efficientnet_b4()

		model.features[0][0] = torch.nn.Conv2d(1, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
		model.classifier = torch.nn.Linear(in_features=1792, out_features=n_classes)

	elif model_name == "whisper":
		model = WhisperModel(n_classes=n_classes, pre_trained=pre_trained) 
	
	else:
		raise ValueError(f"Unrecognized model_name='{model_name}'. Choose from ['efficientnet_b4', 'whisper']")


	return model