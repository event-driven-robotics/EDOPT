# object-track-6dof
Return of the Six (DoF Tracking of Objects)


Assumes a known object model

Three simultaneous computations:

* EROS
* model projection
* state estimation

Build the docker using:

```
cd object-track-6dof
docker build -t sixdof:latest --ssh default --build-arg ssh_pub_key="$(cat ~/.ssh/<publicKeyFile>.pub)" --build-arg ssh_prv_key="$(cat ~/.ssh/<privateKeyFile>)" - < Dockerfile
```
Make the container using:

```
docker run -it -privileged -v /dev/bus/usb:/dev/bus/usb -v /tmp/.X11-unix/:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY --network host --gpus all --name sixdofdev sixdof:latest
```
