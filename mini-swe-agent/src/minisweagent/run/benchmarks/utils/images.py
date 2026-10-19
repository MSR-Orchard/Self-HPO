"""Container image names for benchmark instances (runners and verifiers)."""


def get_swebench_docker_image_name(instance: dict) -> str:
    image_name = instance.get("image_name", None) or instance.get("docker_image", None)
    if image_name is None:
        iid = instance["instance_id"]
        id_docker_compatible = iid.replace("__", "_1776_")
        image_name = f"mirror.gcr.io/swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
    if image_name.startswith("docker.io/"):
        image_name = image_name.replace("docker.io/", "mirror.gcr.io/")
    elif not image_name.startswith("mirror.gcr.io/") and not image_name.startswith("docker://"):
        image_name = "mirror.gcr.io/" + image_name
    return image_name


def get_scaleswe_docker_image_name(instance: dict) -> str:
    image_name = instance.get("image_url") or instance.get("image_name") or instance.get("docker_image")
    if image_name is None:
        raise ValueError(f"No image field found in instance {instance.get('instance_id')}")
    if image_name.startswith("docker.io/"):
        image_name = image_name.replace("docker.io/", "mirror.gcr.io/")
    elif not image_name.startswith("mirror.gcr.io/") and not image_name.startswith("docker://"):
        image_name = "mirror.gcr.io/" + image_name
    return image_name

