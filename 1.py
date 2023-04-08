import wandb

wandb.login()
run = wandb.init(project='1111')

epochs = 10
batches = 5
data = []
for epoch in range(epochs):
    for i in range(batches):
        y = i ** 2
        x = i ** 3
        data.append([x, y])
    table = wandb.Table(data=data, columns=['x', 'y'])
    wandb.log({"yx": wandb.plot.line(table, 'x', 'y', title="xyxyxyxyxy")})