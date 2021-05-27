# Neural-Style-Transfer

![Alt text](https://github.com/yx-chan131/Neural-Style-Transfer/blob/master/images/output.gif?raw=true)


My learning code for Neural Style Transfer which is greatly referenced to [Pytorch Neural Style Transfer](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html#loading-the-images).
In this repo also contains the paper _**A Neural Algorithm of Artistic Style**_ which I hightlighted the text I deemed important. The paper can be downloaded [here](https://arxiv.org/abs/1508.06576).

* **dancing.jpg** as content image

![Alt text](https://github.com/yx-chan131/Neural-Style-Transfer/blob/master/images/dancing.jpg?raw=true)

* **picasso.jpg** as style image

![Alt text](https://github.com/yx-chan131/Neural-Style-Transfer/blob/master/images/picasso.jpg?raw=true)

## Result

**Using content image as input image**

```bash
python main.py # use default setting 
```

Image produced:

![Alt text](https://github.com/yx-chan131/Neural-Style-Transfer/blob/master/images/output_img.png?raw=true)

**Using noise as input image**

```bash
python .\main.py --use_noise True --content_weight 1 --num_steps 1000 --save_anim True --output_path 'images\output_img_from_noise.png'
```

Image produced:

![Alt text](https://github.com/yx-chan131/Neural-Style-Transfer/blob/master/images/output_img_from_noise.png?raw=true)
