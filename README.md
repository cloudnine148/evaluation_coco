# evaluation_coco

## Software Requirement
* Windows or Linux or OSX
* Pyt testResult.chon 3.7 or above
## Usage
### Command
>./python detection_evaluation.py --target <target_name>

Parameter explain:
* The path where the "json" file you want to evaluate is located. <br/>

```
Your project
│   README.md
│   ...
│   detection_evaluate.py
│   ./json/
│        │   ./target_name(1)/
│        │   ./target_name(2)/
│        │   ...
│        └──
└── ./data/
         │   <yolo_format_gt.txt>
         │   ...
         └───
```
### Installation

> pip install -r requirements.txt

### Result
The following result file is created.
> <target_name> testResult.csv
