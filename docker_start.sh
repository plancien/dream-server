#!/bin/bash
echo "Container Started"
echo "conda activate ldm" >> ~/.bashrc
source ~/.bashrc

#cd /workspace/stable-diffusion
#python /workspace/stable-diffusion/scripts/relauncher.py &

echo "RUNPOD start.sh starting"


redis-server &
echo "REDIS started"

if [[ $PUBLIC_KEY ]]
then
    mkdir -p ~/.ssh
    chmod 700 ~/.ssh
    cd ~/.ssh
    echo $PUBLIC_KEY >> authorized_keys
    chmod 700 -R ~/.ssh
    cd /
    service ssh start
    echo "SSH Service Started"
fi


cd /workspace/dream-server
gunicorn -w 4 app:app -b 0.0.0.0:3754  --error-logfile gunicorn.log --access-logfile access.log --access-logformat '%(h)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" %(M)s ms' &
celery -A tasks worker -P solo --loglevel=INFO &
echo "DREAM Server Started, version 0.1.0"


if [[ $JUPYTER_PASSWORD ]]
then
    ln -sf /examples /workspace
    ln -sf /root/welcome.ipynb /workspace

    cd /
    jupyter lab --allow-root --no-browser --port=8888 --ip=* \
        --ServerApp.terminado_settings='{"shell_command":["/bin/bash"]}' \
        --ServerApp.token=$JUPYTER_PASSWORD --ServerApp.allow_origin=* --ServerApp.preferred_dir=/workspace
    echo "Jupyter Lab Started"
fi



sleep infinity