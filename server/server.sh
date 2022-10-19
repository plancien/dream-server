cd /workspace/dream-server
gunicorn -w 4 app:app -b 0.0.0.0:3754  --error-logfile gunicorn.log --access-logfile access.log --access-logformat '%(h)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" %(M)s ms'
