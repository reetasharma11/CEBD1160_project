#### Download this project to your machine 

```
git clone https://github.com/mtl-lan/1160_TeamProject.git
```
#### Locate into the cloned folder 
```
cd 1160_TeamProject
```

#### To build docker image:

```
docker build --tag students_performance:v1 . 
```

#### To create a container from this image:
-v ${PWD}/figures:/app/plots is to mount the plots from the Docker Container to the Host, since once python script finished running, the container will be exited automatically. 

```
docker run -v ${PWD}/figures:/app/plots --name StudentsPerformance students_performance:v1
```