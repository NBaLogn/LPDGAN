import albumentations as albu


def get_transforms(size):
    augs = {'weak': albu.Compose([albu.HorizontalFlip(),
                                  ]),
            'geometric': albu.OneOf([albu.HorizontalFlip(),
                                     albu.ShiftScaleRotate(),
                                     albu.Transpose(),
                                     albu.OpticalDistortion(),
                                     albu.ElasticTransform(),
                                     ])
            }

    aug_fn = augs['geometric']
    crop_fn = {'random': albu.RandomCrop(height=size[0], width=size[1]),
               'center': albu.CenterCrop(height=size[0], width=size[1])}['random']

    effect = albu.OneOf([albu.MotionBlur(blur_limit=21),
                         albu.RandomRain(),
                         albu.RandomFog(),
                         albu.RandomSnow()])
    motion_blur = albu.MotionBlur(blur_limit=55)

    resize = albu.Resize(height=size[0], width=size[1])

    pipeline = albu.Compose([resize], additional_targets={'target': 'image'})

    pipforblur = albu.Compose([effect])

    def process(a, b):
        f = pipforblur(image=a)
        r = pipeline(image=f['image'], target=b)
        return r['image'], r['target']

    return process


def get_transforms_fortest(size):
    resize = albu.Resize(height=size[0], width=size[1])

    effect = albu.OneOf([albu.MotionBlur(),
                         albu.RandomRain(),
                         albu.RandomFog(),
                         albu.RandomSnow()])
    motion_blur = albu.MotionBlur(blur_limit=51)

    pipeline = albu.Compose([resize], additional_targets={'target': 'image'})

    def process(a, b):
        r = pipeline(image=a, target=b)
        return r['image'], r['target']

    return process


def get_normalize():
    normalize = albu.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    normalize = albu.Compose([normalize], additional_targets={'target': 'image'})

    def process(a, b):
        r = normalize(image=a, target=b)
        return r['image'], r['target']

    return process
