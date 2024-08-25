import importlib.util
import logging
import sys
import albumentations as A
import tempfile


def import_module_from_path(module_name, path):
    sys.path.append('/'.join(path.split('/')[:-1]))
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_transforms(img_size, transform_path=None, transforms_string=None):
    if transforms_string and not transform_path:
        logging.info("Loading transforms from temporary file.")
        with tempfile.NamedTemporaryFile(delete=False, suffix='.py', mode='w') as tf:
            tf.write(transforms_string)
            tf_path = tf.name
        transform_path = tf_path
    else:
        logging.info("Loading transforms from provided transform file path.")

    if transform_path:
        transform_module = import_module_from_path(module_name='transform_module', path=transform_path)

        for transform_type in [transform_module.train_transforms, transform_module.val_transforms]:
            for transform in transform_type:
                if hasattr(transform, 'height'):
                    transform.height = img_size
                if hasattr(transform, 'width'):
                    transform.width = img_size
                if hasattr(transform, 'size'):
                    transform.size = (img_size, img_size)

        return transform_module.train_transforms, transform_module.val_transforms

    raise ValueError("Either transform_path or transforms_string must be provided.")
