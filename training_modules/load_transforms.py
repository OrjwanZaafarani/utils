import torchvision.transforms as transforms

# +
transform_low_class1 = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])

transform_low_class2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.5, 1.5), value=0, inplace=False)])

transform_low_class3 = transforms.Compose([
    transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75), shear=(0.1, 0.6)),
    transforms.ToTensor()
])

transform_medium_class1 = transforms.Compose([
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)), 
    transforms.ToTensor()
])

transform_all_classes1 = transforms.Compose([
    transforms.ColorJitter(brightness=(0.6,0.9), contrast=(0.5,0.8), saturation=(0.7,0.9)),
    transforms.ToTensor()
])

transform_all_classes2 = transforms.Compose([
    transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
    transforms.ToTensor()
])

transform_all_classes3 = transforms.Compose([
    transforms.RandomRotation(degrees=(0, 180)),
    transforms.ToTensor()
])

transform_all_classes4 = transforms.Compose([
    transforms.ToTensor(),
    transforms.FiveCrop(size=(200, 200)),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
])


transforms = [transform_low_class1, transform_low_class2, transform_low_class3, transform_medium_class1, transform_all_classes1, transform_all_classes2, transform_all_classes3]
# -















#




