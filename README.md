# EDOPT: Event-camera 6-DoF Dynamic Object Pose Tracking

Paper: https://ieeexplore.ieee.org/abstract/document/10611511

Datasets: https://zenodo.org/records/10829647

```
@inproceedings{glover2024edopt,
  title={EDOPT: Event-camera 6-DoF Dynamic Object Pose Tracking},
  author={Glover, Arren and Gava, Luna and Li, Zhichao and Bartolozzi, Chiara},
  booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={18200--18206},
  year={2024},
  organization={IEEE}
}
```

Assumes a known object model

Three simultaneous computations:

* EROS
* model projection
* state estimation

### Build the docker using:

```
cd EDOPT
docker build -t edopt:latest .

## if you want to build another remote branch for debugging ...
docker build -t edopt:latest --build-arg GIT_BRANCH=your/specified/remote/branch .
```
### Make and enter the container using:

```
docker run -it --privileged -v /dev/bus/usb:/dev/bus/usb -v /tmp/.X11-unix/:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY --network host --gpus all --name edopt edopt:latest
```
or
```
docker compose up -d
docker exec -it edopt /bin/bash
```
### How to build EDOPT
Terminal 1 (on docker container)
```
mkdir -p /usr/local/src/EDOPT/code/build
cd /usr/local/src/EDOPT/code/build
cmake ..
make
```

### How to run EDOPT
Terminal 1 (on docker container)
```
yarpserver
```

Terminal 2 (on docker container)
```
## if you do not have yarp config
yarp config {YOUR YARP IPADDRESS} {PORT}
yarp namespace {NAMESPACE}
yarp detect --write
## Run atis-bridge-sdk to receive event stream
atis-bridge-sdk --s 50
```

Terminal 3 (on docker container)
```
cd /usr/local/src/EDOPT/code/build
./edopt
```

### For development from docker container
Solution for Git Authentication

Terminal 1 (on docker container)
```
gh auth login
```
- Then, select as shown below
  - ? Where do you use GitHub? > GitHub.com
  - ? What is your preferred protocol for Git operations on this host? > HTTPS
  - ? How would you like to authenticate GitHub CLI? > Login with a web browser
  - ! First copy your one-time code: XXXX-XXX
  - Open this link [https://github.com/login/device](https://github.com/login/device)
  - In the web browser, input one-time code to login