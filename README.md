# EDOPT: Event-camera 6-DoF Dynamic Object Pose Tracking

Datasets: https://zenodo.org/records/10829647

Assumes a known object model

Three simultaneous computations:

* EROS
* model projection
* state estimation

### Build the docker using:

```
cd EDOPT
eval $(ssh-agent -s)
ssh-add path/to/your/ssh/secret/key
docker build -t sixdof:latest --ssh default .

## if you want to build another remote branch for debugging ...
docker build -t sixdof:latest --build-arg GIT_BRANCH=your/specified/remote/branch --ssh default  .
```
### Make and enter the container using:

```
docker run -it --privileged -v /dev/bus/usb:/dev/bus/usb -v /tmp/.X11-unix/:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY --network host --gpus all --name sixdofdev sixdof:latest
```
or
```
docker compose up -d
docker exec -it edopt-sixdofdev-1 /bin/bash
```

### How to run EDOPT
Terminal 1 (on host)
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
./sixdofdev
```