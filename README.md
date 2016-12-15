# Academic Lecture Recording Segmentation

## Authors:
* Chengyan Qi (chengyqi)
* Eddie Ye (edwardye)
* Ryan Cesiel (ryances)
* Yunke Cao (ykcao)

## Segmentation
We provide five different versions of TextTiling to run on different lectures, including:
- TextTiling with optimal parameters
- TextTiling with optimal parameters & Cue word Filter
- TextTiling with optimal parameters & Noun Phrase Feature
- TextTiling with optimal parameters & Verbs Feature
- TextTiling with optimal parameters & N-Gram Feature

If you only want to use TextTiling with optimal parameters, then just run
```sh
$ python lecture_segmentation.py  <filename>
```
For example,
```sh
$ python lecture_segmentation.py  data/Engineering_Dynamics/EngDyn_15
```
If you want to include extra features, the usage is
```sh
$ python lecture_segmentation.py  <filename> <Extra_feature>[Optional]
```
For example,
```sh
$ python lecture_segmentation.py  data/Engineering_Dynamics/EngDyn_15 Verb
```
If you want to set the weight of the new feature, the usage is
```sh
$ python lecture_segmentation.py  <filename> <Extra_feature>[Optional] <feature_weight>[Optional]
```
For example,
```sh
$ python lecture_segmentation.py  data/Engineering_Dynamics/EngDyn_15 Verb 0.2
```

## Web Application Prototype
The web application is contained within `web/` folder.

### Installation
First install dependencies using our requirements.txt file:
```sh
pip install -r requirements.txt
```

### Deployment
To run our website locally:
```sh
python run.py
```

This will start the website locally on port 5000. You can visit localhost:5000 and query the second lecture from the ["Introduction to Psychology" course](https://ocw.mit.edu/courses/brain-and-cognitive-sciences/9-00sc-introduction-to-psychology-fall-2011/brain-i/).
