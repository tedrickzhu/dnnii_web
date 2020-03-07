#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 19-11-29 下午2:37
# @Author  : zhuzhengyi
# @File    : searcher.py
# @Software: PyCharm
import numpy
import csv
import re


class Searcher:
	# colorIndexPath色彩空间特征索引表路径，structureIndexPath结构特征索引表路径
	__slot__ = ["colorIndexPath", "structureIndexPath"]

	def __init__(self, colorIndexPath, structureIndexPath):
		self.colorIndexPath, self.structureIndexPath = colorIndexPath, structureIndexPath

	# 计算色彩空间的距离，卡方相似度计算
	def solveColorDistance(self, features, queryFeatures, eps=1e-5):
		distance = 0.5 * numpy.sum([((a - b) ** 2) / (a + b + eps) for a, b in zip(features, queryFeatures)])
		# distance = 0.5 * numpy.sum([((a - b) ** 2) / (a + b + eps) for a, b in zip(features, numpy.array(queryFeatures))])
		return distance

	# 计算构图空间的距离
	def solveStructureDistance(self, structures, queryStructures, eps=1e-5):
		distance = 0
		normalizeRatio = 5e3
		for index in range(len(queryStructures)):
			for subIndex in range(len(queryStructures[index])):
				a = structures[index][subIndex]
				b = queryStructures[index][subIndex]
				distance += (a - b) ** 2 / (a + b + eps)
		return distance / normalizeRatio

	def searchByColor(self, queryFeatures):
		searchResults = {}
		with open(self.colorIndexPath) as indexFile:
			reader = csv.reader(indexFile)
			for line in reader:
				features = []
				for feature in line[1:]:
					feature = feature.replace("[", "").replace("]", "")
					findStartPosition = 0
					feature = re.split("\s+", feature)
					rmlist = []
					for index, strValue in enumerate(feature):
						if strValue == "":
							rmlist.append(index)
					for _ in range(len(rmlist)):
						currentIndex = rmlist[-1]
						rmlist.pop()
						del feature[currentIndex]
					feature = [float(eachValue) for eachValue in feature]
					features.append(feature)
				distance = self.solveColorDistance(features, queryFeatures)
				searchResults[line[0]] = distance
			indexFile.close()
		# print "feature", sorted(searchResults.iteritems(), key = lambda item: item[1], reverse = False)
		return searchResults

	def transformRawQuery(self, rawQueryStructures):
		queryStructures = []
		for substructure in rawQueryStructures:
			structure = []
			for line in substructure:
				for tripleColor in line:
					structure.append(float(tripleColor))
			queryStructures.append(structure)
		return queryStructures

	def searchByStructure(self, rawQueryStructures):
		searchResults = {}
		queryStructures = self.transformRawQuery(rawQueryStructures)
		# queryStructures = rawQueryStructures
		with open(self.structureIndexPath) as indexFile:
			reader = csv.reader(indexFile)
			for line in reader:
				structures = []
				for structure in line[1:]:
					structure = structure.replace("[", "").replace("]", "")
					structure = re.split("\s+", structure)
					if structure[0] == "":
						structure = structure[1:]
					structure = [float(eachValue) for eachValue in structure]
					structures.append(structure)
				distance = self.solveStructureDistance(structures, queryStructures)
				searchResults[line[0]] = distance
			indexFile.close()
		# print "structure", sorted(searchResults.iteritems(), key = lambda item: item[1], reverse = False)
		return searchResults

	def search(self, queryFeatures, rawQueryStructures, limit=3):
		featureResults = self.searchByColor(queryFeatures)
		structureResults = self.searchByStructure(rawQueryStructures)
		resultsdict = {}
		for key, value in featureResults.items():
			resultsdict[key] = value + structureResults[key]

		# results = sorted(resultsdict.items(), key=lambda item: item[1], reverse=False)
		results = []
		for circle in range(limit):
			maxscore = None
			maxscorekey = None
			for key,score in resultsdict.items():
				if (maxscore is None) or (score>maxscore):
					maxscore = score
					maxscorekey = key
			results.append([maxscorekey,maxscore])
			resultsdict.pop(maxscorekey)

		return results