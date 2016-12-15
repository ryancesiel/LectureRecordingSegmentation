# Academic Lecture Recording Segmentation

## Authors:
* Chengyan Qi (chengyqi)
* Eddie Ye (edwardye)
* Ryan Cesiel (ryances)
* Yunke Cao (ykcao)

## File Information
* `/`: contains all files for TextTiling, our different features, and the evaluation of these files
 * The "Segmentation Models" section below describes running files within this folder.

* `/data`: this folder contains our annotated data. Below are links to the unnannotated data:
 * [Engineering Dynamics (Fall 2011)](https://ocw.mit.edu/courses/mechanical-engineering/2-003sc-engineering-dynamics-fall-2011/)
 * [Introduction to Psychology (Fall 2011)](https://ocw.mit.edu/courses/brain-and-cognitive-sciences/9-00sc-introduction-to-psychology-fall-2011/)
 * [Principles of Microeconomics (Fall 2011)](https://ocw.mit.edu/courses/economics/14-01sc-principles-of-microeconomics-fall-2011/)

* `/results`: this folder contains data on testing different TextTiling parameters, testing the weighting scheme of our different features, output from automatic speech recognition systems, and cue word research.

* `/web`: this folder contains the information retrieval web app prototype of our system.
 * The "Web Application Prototype" section below describes running this application.

## Segmentation Models
We provide five different versions of TextTiling to run on different lectures, including:
- TextTiling with optimal parameters
- TextTiling with optimal parameters & Cue word Filter
- TextTiling with optimal parameters & Noun Phrase Feature
- TextTiling with optimal parameters & Verbs Feature
- TextTiling with optimal parameters & N-Gram Feature

## Dependency Installation
First install dependencies using our requirements.txt file:
```sh
pip install -r requirements.txt
```

## Running Different Models
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

## Acknowledgements
In addition to our "Related Works" section of our final report, our software relies heavily on the NLP open-source libraries:
* NLTK
* TextBlob
