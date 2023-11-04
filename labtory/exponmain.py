import yaml

with open("model_config.yaml") as file:
    cfg= yaml.safe_load(file)

print(dict(cfg))