# Lecture Recording Segmentation
This project is ...

We provide five different versions of TextTiling to run on different lectures, includes:
- TextTiling with opitimal parameters
- TextTiling with opitimal parameters & Cue word Filter
- TextTiling with opitimal parameters & Noun Phrase Feature
- TextTiling with opitimal parameters & Verbs Feature
- TextTiling with opitimal parameters & N-Gram Feature

If you only want to use TextTiling with opitimal parameters, then just run
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