source activate dnnii_web_gpu
nohup python -u manage.py runserver -h 0.0.0.0 -p 5555 >output.log 2>&1 &