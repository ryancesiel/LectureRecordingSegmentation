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
		'end': '00:14:28'
	},
	'2': {
		'start': '00:14:28',
		'end': '00:17:35'
	},
	'3': {
		'start': '00:17:35',
		'end': '00:45:36'
	},
	'4': {
		'start': '00:45:36',
		'end': '00:49:19'
	},
	'5': {
		'start': '00:49:19',
		'end': '00:50:47'
	},
	'6': {
		'start': '00:50:47',
		'end': '00:54:14'
	},
	'7': {
		'start': '00:54:14',
		'end': '00:59:48'
	},
	'8': {
		'start': '00:59:48',
		'end': '01:01:13'
	},
	'9': {
		'start': '01:01:13',
		'end': '01:10:05'
	},
	'10': {
		'start': '01:10:05',
		'end': ','
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
			if not len(data['results']):
				data['error'] = True

			return render_template('index.html', **data)

	return render_template('index.html', **data)

@app.route('/segment/<segment_id>')
def segment(segment_id):
	data = {}

	data['start'] = TIMES[segment_id]['start']
	data['end'] = TIMES[segment_id]['end']
	index = open_dir("index")
	with index.searcher() as searcher:
		query = QueryParser("segment_id", index.schema).parse(segment_id)
		results = searcher.search(query)
		data['transcript'] = results[0]['transcript']
		return render_template('segment.html', **data)
