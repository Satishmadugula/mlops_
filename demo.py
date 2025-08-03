import yaml

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

project_name = config.get('Project', 'Unknown Project')
docker_images = config.get('Docker', {}).get('Docker_Imges', 'No Docker Images specified')

print(f"Project Name: {project_name}")
print(f"Docker Images: {docker_images}")