package lda;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;

public class Main
{
	public static final int TYPE = 10;
	public static final int NUMBER = 6;
	public static final int SUM = 2;
	public static final int RANK = 6;
	
	public static final double omega = 0.005;
	public static final double eta = 0.01;
	public static final double delta = 0.25;
	
	public static final int top_n_service = 1;  
	public static final int top_n_word = RANK;

	public static String[][][] info = new String[TYPE][NUMBER][SUM];

	public static ArrayList<String[][]> array = new ArrayList<>();
	public static Scanner scanner = new Scanner(System.in);

	public static void main(String[] args) throws IOException
	{
		Corpus.TYPE = TYPE;
		Corpus.NUMBER = NUMBER;
		Corpus.SUM = SUM;
		Corpus.RANK = RANK;
		Corpus.OMEGA = omega;
		Corpus.DELTA = delta;
		Corpus.ETA = eta;
		Corpus.TOP_N_SERVICE = top_n_service;

		// Corpus corpus =
		// Corpus.load("/home/elvis/work/java/ManagerLDA/data/mini");
		
		Corpus corpus = Corpus.load("/home/elvis/work/SR/content/");
		
		LdaGibbsSampler ldaGibbsSampler = new LdaGibbsSampler(corpus.getDocument(), corpus.getVocabularySize());

		ldaGibbsSampler.gibbs(TYPE);

		double[][] phi = ldaGibbsSampler.getPhi();

		Map<String, Double>[] topicMap = LdaUtil.translate(phi, corpus.getVocabulary(), NUMBER);

		Main.show(topicMap);
		
		// Map<String, Double> allMap = new HashMap();
		Set<String> allTopics = new HashSet();
		
		for(int i = 0; i < topicMap.length; i++)
		{
			Map<String, Double> temp = topicMap[i];
			for (Map.Entry<String, Double> entry : temp.entrySet())
			{
				String word = entry.getKey();
				double value = entry.getValue();
				allTopics.add(word);
			}
		}
		
		double sum = 0;

		for (int i = 0; i < TYPE; i++)
		{
			System.out.println("Topic " + i);
			for (int j = 0; j < NUMBER; j++)
			{
				for (int k = 0; k < SUM; k++)
				{
					System.out.println(info[i][j][k]);
					if(k == 1)
					{
						sum += Double.parseDouble(info[i][j][k]);
					}
				}
			}
			System.out.println();
		}

		System.out.println("sum: " + sum);
		
		// System.exit(-145);
		
		/*
		String topic = info[3][3][0];
		System.out.println("TOPIC : " + topic);
		System.out.println("TOPIC FIND FILE");
		LinkedList<String> rearchFile = topicFindFile(topic);
		for (int i = 0; i < rearchFile.size(); i++)
		{
			System.out.print(rearchFile.get(i) + "\t");
		}
		System.out.println();

		System.out.println("TOPIC FIND TOPIC");
		LinkedList<String> rearchTopic = topicFindTopic(info[3][3][0]);
		for (int i = 0; i < rearchTopic.size(); i++)
		{
			System.out.print(rearchTopic.get(i) + "\t");
		}
		
		System.out.println();
		System.out.println("FILE FIND FILE");
		LinkedList<String> fileReasearch = fileFindFIle(corpus.documentName.get(3));
		for (int i = 0; i < fileReasearch.size(); i++)
		{
			System.out.print(fileReasearch.get(i) + "\t");
		}
		
		System.out.println();
		System.out.println("FILE FIND TOPIC");
		LinkedList<String> topicReasearch = fileFindTopic("0060914882.txt");
		for (int i = 0; i < topicReasearch.size(); i++)
		{
			System.out.print(topicReasearch.get(i) + "\t");
		}
		System.out.println();
		*/
		
		double[][] doc2TopicMatrix = ldaGibbsSampler.getTheta();
		double[][] topic2WordMatrix = ldaGibbsSampler.getPhi();
		// double[][] word2topicMatrix = ldaGibbsSampler.getPhi();
		// System.out.println(doc2TopicMatrix.length);
		// System.out.println(corpus.documentName.size());
		
		
		// System.exit(0);
		
		String thetaFileName = "/home/elvis/work/SR/doc2topics_matrix.txt";
		String phiFileName = "/home/elvis/work/SR/word2topic_matrix.txt";
		// String docTopicsFileName = "/home/elvis/work/SR/doc2topics.txt";
		String serviceTopicFolder = "/home/elvis/work/SR/service_topic/";
		
		// saveTopic
		
		saveThetaMatrix(thetaFileName, doc2TopicMatrix, Corpus.documentName);
		savePhiMatrix(phiFileName, info);
		saveDocTopics(Corpus.documentName, serviceTopicFolder, allTopics);

		// saveDocTopics(docTopicsFileName, );
		
	}

	private static void saveDocTopics(List<String> documentName, String serviceTopicFolder, Set<String> allTopics)
	{
		// TODO Auto-generated method stub
		for(int index = 0; index < Corpus.documentName.size(); index++)
		{
			String fileName = Corpus.documentName.get(index);
			LinkedList<String> result = fileFindTopic(fileName);
			
			// System.out.println(fileName);
			String content = "";
			for(int i = 0; i < result.size(); i++)
			{
				// System.out.println(result.get(i));
				String word = result.get(i);
				// System.out.println(fileName);
				
				if(isInTopic(word, allTopics))
				{
					content += word + "\n";
				}
			}
			writeInTopicFile(serviceTopicFolder, fileName, content);
		}

	}

	private static void savePhiMatrix(String phiFileName, String[][][] info)
	{
		// TODO Auto-generated method stub

		
		FileWriter fw;
		try
		{
			fw = new FileWriter(phiFileName);
			
			Set<String> nameSet = new HashSet<>();

			for (int i = 0; i < TYPE; i++)
			{
				for (int j = 0; j < NUMBER; j++)
				{
					for (int k = 0; k < SUM; k++)
					{
						
						// System.out.println(info[i][j][k]);
						if(k == 1)
						{
							// sum += Double.parseDouble(info[i][j][k]);
						}
						if(k == 0)
						{
							nameSet.add(info[i][j][k]);
						}
					}
				}
				// System.out.println();
			}
			
			double[][] word2topicMatrix = new double[nameSet.size()][TYPE];
			String[] nameList = new String[nameSet.size()];
			
			int index = 0;
			
			for(String name: nameSet)
			{
				nameList[index] = name;
				index ++;
			}
			
			for (int i = 0; i < TYPE; i++)
			{
				for (int j = 0; j < NUMBER; j++)
				{
					
					String word = info[i][j][0];
					double value = Double.parseDouble(info[i][j][1]);
					// System.out.println(value);
					
					int choose = -1;
					
					for(int k = 0; k < nameList.length; k++)
					{
						if(nameList[k].hashCode() == word.hashCode())
						{
							choose = k;
							break;
						}
					}
					
					word2topicMatrix[choose][i] = value;
					
				}
				// System.out.println();
			}			

			String content = "";
			
			for(int i = 0; i < word2topicMatrix.length; i++)
			{
				// System.out.print(nameList[i] + "\t");
				content += nameList[i];
				for(int k = 0; k < word2topicMatrix[i].length; k++)
				{
					// System.out.print(word2topicMatrix[i][k] + "\t");
					content += ","  + word2topicMatrix[i][k];	
				}
				content += "\n";
				// System.out.println();
			}
			
			fw.write(content);
			
			fw.close();
		} catch (IOException e)
		{
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.exit(-4);
		}		
		// System.exit(-19);
	}

	private static boolean isInTopic(String word, Set<String> allTopics)
	{
		// TODO Auto-generated method stub
		return allTopics.contains(word);
	}

	private static void writeInTopicFile(String serviceTopicFolder, String fileName, String content)
	{
		// TODO Auto-generated method stub
		FileWriter fw;
		try
		{
			fw = new FileWriter(serviceTopicFolder + fileName);
			fw.write(content);
			fw.close();
		} catch (IOException e)
		{
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.exit(-5);
		}		
	}

	private static void saveThetaMatrix(String thetaFileName, double[][] doc2TopicMatrix, List<String> documentName)
	{
		// TODO Auto-generated method stub
		if(doc2TopicMatrix.length != documentName.size())
		{
			System.err.println("ERROR FOR SIZE CHECK");
			System.exit(-14);
		}
		
		
		FileWriter fw;
		try
		{
			fw = new FileWriter(thetaFileName);
			
			for(int i = 0; i < documentName.size(); i++)
			{
				String line = documentName.get(i);
				String content = "";
				for(int j = 0; j < doc2TopicMatrix[i].length; j++)
				{
					content += "," + doc2TopicMatrix[i][j];
				}
				
				line += content + "\n";
				
				fw.write(line);
			}			
			
			fw.close();
		} catch (IOException e)
		{
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.exit(-4);
		}
	}

	public static LinkedList<String> fileFindFile(String file)
	{
		LinkedList<String> result = new LinkedList<>();
		LinkedList<String> search = fileFindTopic(file);
		for (int i = 0; i < search.size(); i++)
		{
			String topic = search.get(i);
			LinkedList<String> fileSet = topicFindFile(topic);
			for (int j = 0; j < fileSet.size(); j++)
			{
				String fileName = fileSet.get(j);
				result.add(fileName);
			}
		}
		return result;
	}

	public static LinkedList<String> topicFindTopic(String topic)
	{
		LinkedList<String> result = new LinkedList<>();
		int hash = topic.hashCode();
		for (int i = 0; i < TYPE; i++)
		{
			for (int j = 0; j < NUMBER; j++)
			{
				String word = info[i][j][0];
				if (word == null)
				{
					continue;
				}
				if (hash == info[i][j][0].hashCode())
				{
					for (int k = 0; k < NUMBER; k++)
					{
						result.add(info[i][k][0]);
					}
					return result;
				}
			}
		}
		return result;
	}

	public static LinkedList<String> topicFindFile(String topic)
	{
		LinkedList<String> result = new LinkedList<>();
		int topicHash = topic.hashCode();
		for (int i = 0; i < Corpus.documentName.size(); i++)
		{
			for (int j = 0; j < Corpus.topicWord.get(i).length; j++)
			{
				String word = Corpus.topicWord.get(i)[j];
				if (word == null)
				{
					continue;
				}
				if (word.hashCode() == topicHash)
				{
					String fileName = Corpus.documentName.get(i);
					result.add(fileName);
				}
			}
		}
		return result;
	}

	public static LinkedList<String> fileFindTopic(String fileName)
	{
		LinkedList<String> result = new LinkedList<>();
		int hash = fileName.hashCode();
		for (int i = 0; i < Corpus.documentName.size(); i++)
		{
			String docName = Corpus.documentName.get(i);
			if (docName == null)
			{
				continue;
			}
			if (docName.hashCode() == hash)
			{
				for (int j = 0; j < Corpus.topicWord.get(i).length; j++)
				{
					String findTopic = Corpus.topicWord.get(i)[j];
					result.add(findTopic);
				}
				return result;
			}
		}
		return result;
	}

	public static void show(Map<String, Double>[] map)
	{
		int i = 0;
		for (Map<String, Double> topicMap : map)
		{
			int j = 0;
			for (Map.Entry<String, Double> entry : topicMap.entrySet())
			{
				info[i][j][0] = entry.getKey();
				info[i][j][1] = entry.getValue() + "";
				j++;
			}
			i++;
		}
	}
}
