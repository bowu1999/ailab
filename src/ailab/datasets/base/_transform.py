from torchvision import transforms


std_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda image: image.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.48145466, 0.4578275, 0.40821073],
        std = [0.26862954, 0.26130258, 0.27577711]
    )
])