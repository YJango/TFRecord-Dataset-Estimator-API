# TensorFlow API: Estimator

TensorFlow 目前是最为常用的深度学习框架。然而从头编写整个机器学习流程却是一件繁琐的工作。虽然上有 Keras 帮助减轻代码量，但这次 TensorFlow 自己提供了高阶 API estimators。不同的是 estimators 并不注重网络结构的搭建，而是帮助你更有效率的实现机器学习不同阶段的操作。可以在保证网络结构控制权的基础上，节省工作量。这里记录的是如何使用高层API来使用Deep Learning。





{% embed data="{\"url\":\"https://plot.ly/~gxiiukk/52/\",\"type\":\"rich\",\"title\":\"data preprocessing plot \| scatter chart made by Gxiiukk \| plotly\",\"description\":\"Gxiiukk\'s interactive graph and data of \\"data preprocessing plot\\" is a scatter chart, showing original vs standardization; with time in the x-axis and value in the y-axis..\",\"icon\":{\"type\":\"icon\",\"url\":\"https://plot.ly/favicon.ico?v=2\",\"aspectRatio\":0},\"embed\":{\"type\":\"app\",\"html\":\"<a href=\\"https://plot.ly/~gxiiukk/52/\\" target=\\"\_blank\\" style=\\"display: block; text-align: center;\\"><img src=\\"https://plot.ly/~gxiiukk/52.png\\"/></a><script data-plotly=\\"gxiiukk:52\\" src=\\"https://plot.ly/embed.js\\" async></script>\",\"aspectRatio\":0}}" %}



