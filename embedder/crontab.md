

# C24 crontab setup


```
0 7 * * * cd /opt/chat-ai-deployment/embedder/jina && docker-compose restart

1 7 * * * cd /opt/chat-ai-deployment/embedder && docker-compose -f docker-compose-balancer.yaml restart

```