# Playground 

Playground tasks for newcomers!

## 1. Environments

Fork this repository to your own github account:

![image](./docs/imgs/fork.png)

Start a docker from image `playground:v1.0-cuda12.2-cudnn8-ubuntu22`:

```bash
# Params:
#   <container-name>: Your container name. Can be anything.
#   <container-port>: A port that nobody has used. For example: 1145.
#   <project-home>: A directory that stores all your projects. This directory should be shared by all containers.
#   <proxy-addr>: Your proxy address.
#   --cap-add=SYS_ADMIN: This is required for using Nsight Compute to profiling in docker.
docker run --gpus all --name <container-name> -it \
    -p <container-port>:22 --entrypoint /bin/bash \
    --cap-add=SYS_ADMIN \
    -v ~/<project-home>:/root/<project-home>  \
    -e HTTP_PROXY=<proxy-addr>  \
    -e HTTPS_PROXY=<proxy-addr>  \
    -e http_proxy=<proxy-addr>  \
    -e https_proxy=<proxy-addr>  \
    playground:v1.0-cuda12.2-cudnn8-ubuntu22
```

Inside the container, clone the repo you forked:

```bash
cd /root/<project-home>
git clone https://github.com/<your-github-account-name>/playground.git
```

Remember to change your git config:

```bash
git config --global user.name "<your-name>"
git config --global user.email "<your-email>"
```

## 2. Tasks

| No. | Discription | Directory | Document |
|:---:|:---:|:---:|:---:|
| 1 | CUDA Programming: GEMM | [./task-1](./task-1) | [READEME.md](./task-1/README.md) |

