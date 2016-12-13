'''
	Project: 		EECS 498 Fall 2016 Final Project
					Lecture Recording Segmentation

	File:			Web Application Views

	Description:	Defines the pages for the web application.

	Authors:		Chengyan Qi (chengyqi)
					Edward Ye (edwardye)
					Ryan Cesiel (ryances)
                	Yunke Cao (ykcao)
'''

from flask import render_template, request
from app import app

@app.route('/', methods = ['GET', 'POST'])
def index():
	data = {}

	# if request.method == 'POST':


	return render_template('index.html', **data)
