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
from whoosh.qparser import QueryParser
from whoosh.index import open_dir

TIMES = {
	'1': {
		'start': '00:00:00',
		'end': '00:04:53'
	},
	'2': {
		'start': '00:04:54',
		'end': '00:20:10'
	},
	'3': {
		'start': '00:20:11',
		'end': '00:30:00'
	},
	'4': {
		'start': '00:30:01',
		'end': '00:37:12'
	},
	'5': {
		'start': '00:37:13',
		'end': '00:49:16'
	}
}

@app.route('/', methods = ['GET', 'POST'])
def index():
	data = {}

	if request.method == 'POST':
		query_text = request.form['query']

		index = open_dir("index")
		with index.searcher() as searcher:
			query = QueryParser("transcript", index.schema).parse(query_text)
			results = searcher.search(query)
			data['results'] = []
			for result in results:
				data['results'].append({
					'course_name': result['course_name'],
					'segment_id': result['segment_id'],
					'transcript': result['transcript'][:250]
				})

			return render_template('index.html', **data)

	return render_template('index.html', **data)

@app.route('/segment/<segment_id>')
def segment(segment_id):
	data = {}

	data['start'] = TIMES[segment_id]['start']
	data['end'] = TIMES[segment_id]['end']
	return render_template('segment.html', **data)
