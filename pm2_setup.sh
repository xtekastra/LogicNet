pm2 install pm2-logrotate

pm2 set pm2-logrotate:max_size 10M
pm2 set pm2-logrotate:compress false
# Rotate every minute
pm2 set pm2-logrotate:rotateInterval '* * * * *'  
pm2 set pm2-logrotate:retain 24  # Keep 24 rotated files (1 day)