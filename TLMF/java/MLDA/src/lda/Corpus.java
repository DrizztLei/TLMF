/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2015/1/29 17:03</create-date>
 *
 * <copyright file="Corpus.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package lda;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Scanner;

// import java.util.Scanner;

import lda.Vocabulary;

public class Corpus
{
	static List<int[]> documentList;
	public static List<String[]> topicWord;
	public static List<String> documentName;
	public static int count = 0;
	public static int TYPE, NUMBER, SUM, RANK;
	public static HashSet<String> stopWord;

	public static double TOP_N_SERVICE;
	public static double OMEGA, ETA, DELTA;
	private static ArrayList<String> scatterList;

	Vocabulary vocabulary;

	public Corpus()
	{
		documentList = new LinkedList<int[]>();
		documentName = new LinkedList<String>();
		vocabulary = new Vocabulary();
		topicWord = new LinkedList<>();
		stopWord = new HashSet<String>();

		stopWord.add("a");
		stopWord.add("a");
		stopWord.add("able");
		stopWord.add("about");
		stopWord.add("above");
		stopWord.add("according");
		stopWord.add("accordingly");
		stopWord.add("across");
		stopWord.add("actually");
		stopWord.add("after");
		stopWord.add("afterwards");
		stopWord.add("again");
		stopWord.add("against");
		stopWord.add("all");
		stopWord.add("ad");
		stopWord.add("well");
		stopWord.add("arthur");
		stopWord.add("st");
		// stopWord.add("allow");
		// stopWord.add("allows");
		// stopWord.add("almost");
		stopWord.add("alone");
		stopWord.add("along");
		stopWord.add("already");
		stopWord.add("also");
		stopWord.add("although");
		stopWord.add("always");
		stopWord.add("am");
		stopWord.add("among");
		stopWord.add("amongst");
		stopWord.add("an");
		stopWord.add("and");
		stopWord.add("another");
		stopWord.add("any");
		stopWord.add("anybody");
		stopWord.add("anyhow");
		stopWord.add("anyone");
		stopWord.add("anything");
		stopWord.add("anyway");
		stopWord.add("anyways");
		stopWord.add("anywhere");
		stopWord.add("apart");
		stopWord.add("appear");
		// stopWord.add("appreciate");
		stopWord.add("appropriate");
		stopWord.add("are");
		stopWord.add("around");
		stopWord.add("as");
		stopWord.add("aside");
		stopWord.add("ask");
		stopWord.add("asking");
		stopWord.add("associated");
		stopWord.add("at");
		stopWord.add("available");
		stopWord.add("away");
		// stopWord.add("awfully");
		stopWord.add("b");
		stopWord.add("be");
		stopWord.add("became");
		stopWord.add("because");
		stopWord.add("become");
		stopWord.add("becomes");
		stopWord.add("becoming");
		stopWord.add("been");
		stopWord.add("before");
		stopWord.add("beforehand");
		stopWord.add("behind");
		stopWord.add("being");
		stopWord.add("believe");
		stopWord.add("below");
		stopWord.add("beside");
		stopWord.add("besides");
		// stopWord.add("best");
		// stopWord.add("better");
		stopWord.add("between");
		stopWord.add("beyond");
		stopWord.add("both");
		stopWord.add("book");
		stopWord.add("but");
		stopWord.add("brief");
		stopWord.add("by");
		stopWord.add("c");
		stopWord.add("came");
		stopWord.add("can");
		stopWord.add("certain");
		stopWord.add("certainly");
		stopWord.add("clearly");
		stopWord.add("co");
		stopWord.add("com");
		stopWord.add("come");
		stopWord.add("comes");
		stopWord.add("contain");
		stopWord.add("containing");
		stopWord.add("contains");
		stopWord.add("corresponding");
		stopWord.add("could");
		stopWord.add("course");
		stopWord.add("currently");
		stopWord.add("d");
		stopWord.add("definitely");
		stopWord.add("described");
		stopWord.add("despite");
		stopWord.add("did");
		stopWord.add("different");
		stopWord.add("do");
		stopWord.add("does");
		stopWord.add("doing");
		stopWord.add("done");
		stopWord.add("down");
		stopWord.add("downwards");
		stopWord.add("during");
		stopWord.add("e");
		stopWord.add("each");
		stopWord.add("edu");
		stopWord.add("eg");
		stopWord.add("eight");
		stopWord.add("either");
		stopWord.add("else");
		stopWord.add("elsewhere");
		stopWord.add("enough");
		stopWord.add("entirely");
		stopWord.add("especially");
		stopWord.add("et");
		stopWord.add("etc");
		stopWord.add("even");
		stopWord.add("ever");
		stopWord.add("every");
		stopWord.add("everybody");
		stopWord.add("everyone");
		stopWord.add("everything");
		stopWord.add("everywhere");
		stopWord.add("ex");
		stopWord.add("exactly");
		stopWord.add("example");
		stopWord.add("except");
		stopWord.add("f");
		stopWord.add("far");
		stopWord.add("few");
		stopWord.add("fifth");
		stopWord.add("first");
		stopWord.add("five");
		stopWord.add("followed");
		stopWord.add("following");
		stopWord.add("follows");
		stopWord.add("for");
		stopWord.add("former");
		stopWord.add("formerly");
		stopWord.add("forth");
		stopWord.add("four");
		stopWord.add("from");
		stopWord.add("further");
		stopWord.add("furthermore");
		stopWord.add("g");
		stopWord.add("get");
		stopWord.add("gets");
		stopWord.add("getting");
		stopWord.add("given");
		stopWord.add("gives");
		stopWord.add("go");
		stopWord.add("goes");
		stopWord.add("going");
		stopWord.add("gone");
		stopWord.add("got");
		stopWord.add("gotten");
		// stopWord.add("greetings");
		stopWord.add("h");
		stopWord.add("had");
		stopWord.add("happens");
		// stopWord.add("hardly");
		stopWord.add("has");
		stopWord.add("have");
		stopWord.add("having");
		stopWord.add("he");
		stopWord.add("hello");
		stopWord.add("help");
		stopWord.add("hence");
		stopWord.add("her");
		stopWord.add("here");
		stopWord.add("hereafter");
		stopWord.add("hereby");
		stopWord.add("herein");
		stopWord.add("hereupon");
		stopWord.add("hers");
		stopWord.add("herself");
		stopWord.add("hi");
		stopWord.add("him");
		stopWord.add("himself");
		stopWord.add("his");
		stopWord.add("hither");
		// stopWord.add("hopefully");
		stopWord.add("how");
		stopWord.add("howbeit");
		stopWord.add("however");
		stopWord.add("i");
		stopWord.add("ie");
		stopWord.add("if");
		// stopWord.add("ignored");
		stopWord.add("immediate");
		stopWord.add("in");
		stopWord.add("inasmuch");
		stopWord.add("inc");
		stopWord.add("indeed");
		stopWord.add("indicate");
		stopWord.add("indicated");
		stopWord.add("indicates");
		stopWord.add("inner");
		stopWord.add("insofar");
		stopWord.add("instead");
		stopWord.add("into");
		stopWord.add("inward");
		stopWord.add("is");
		stopWord.add("it");
		stopWord.add("its");
		stopWord.add("itself");
		stopWord.add("j");
		stopWord.add("just");
		stopWord.add("k");
		stopWord.add("keep");
		stopWord.add("keeps");
		stopWord.add("kept");
		stopWord.add("know");
		stopWord.add("knows");
		stopWord.add("known");
		stopWord.add("l");
		stopWord.add("last");
		stopWord.add("lately");
		stopWord.add("later");
		stopWord.add("latter");
		stopWord.add("latterly");
		stopWord.add("least");
		stopWord.add("less");
		stopWord.add("lest");
		stopWord.add("let");
		stopWord.add("like");
		stopWord.add("liked");
		stopWord.add("likely");
		stopWord.add("little");
		stopWord.add("ll"); // added to avoid words like you'll,I'll etc.
		stopWord.add("look");
		stopWord.add("looking");
		stopWord.add("looks");
		stopWord.add("ltd");
		stopWord.add("m");
		stopWord.add("mainly");
		stopWord.add("many");
		stopWord.add("may");
		stopWord.add("maybe");
		stopWord.add("me");
		// stopWord.add("mean");
		stopWord.add("meanwhile");
		// stopWord.add("merely");
		stopWord.add("might");
		stopWord.add("more");
		stopWord.add("moreover");
		stopWord.add("most");
		stopWord.add("make");
		stopWord.add("mostly");
		stopWord.add("much");
		stopWord.add("must");
		stopWord.add("my");
		stopWord.add("myself");
		stopWord.add("n");
		stopWord.add("name");
		stopWord.add("namely");
		stopWord.add("nd");
		stopWord.add("near");
		stopWord.add("nearly");
		stopWord.add("necessary");
		stopWord.add("need");
		stopWord.add("needs");
		// stopWord.add("neither");
		// stopWord.add("never");
		// stopWord.add("nevertheless");
		stopWord.add("new");
		stopWord.add("next");
		stopWord.add("nine");
		stopWord.add("normally");
		// stopWord.add("novel");
		stopWord.add("no");
		stopWord.add("nobody");
		stopWord.add("non");
		stopWord.add("none");
		stopWord.add("noone");
		stopWord.add("nor");
		stopWord.add("normally");
		stopWord.add("not");
		stopWord.add("n't");
		stopWord.add("nothing");
		stopWord.add("novel");
		stopWord.add("now");
		stopWord.add("nowhere");
		stopWord.add("now");
		stopWord.add("nowhere");
		stopWord.add("o");
		stopWord.add("obviously");
		stopWord.add("of");
		stopWord.add("off");
		stopWord.add("often");
		stopWord.add("oh");
		stopWord.add("ok");
		stopWord.add("okay");
		// stopWord.add("old");
		stopWord.add("on");
		stopWord.add("once");
		stopWord.add("one");
		stopWord.add("ones");
		stopWord.add("only");
		stopWord.add("onto");
		stopWord.add("or");
		stopWord.add("other");
		stopWord.add("others");
		stopWord.add("otherwise");
		stopWord.add("ought");
		stopWord.add("our");
		stopWord.add("ours");
		stopWord.add("ourselves");
		stopWord.add("out");
		stopWord.add("outside");
		stopWord.add("over");
		stopWord.add("overall");
		stopWord.add("own");
		stopWord.add("p");
		stopWord.add("particular");
		stopWord.add("particularly");
		stopWord.add("per");
		stopWord.add("perhaps");
		stopWord.add("placed");
		stopWord.add("please");
		stopWord.add("plus");
		stopWord.add("possible");
		stopWord.add("presumably");
		stopWord.add("probably");
		stopWord.add("provides");
		stopWord.add("q");
		stopWord.add("que");
		stopWord.add("quite");
		stopWord.add("qv");
		stopWord.add("r");
		stopWord.add("rather");
		stopWord.add("rd");
		stopWord.add("re");
		stopWord.add("really");
		stopWord.add("reasonably");
		stopWord.add("regarding");
		stopWord.add("regardless");
		stopWord.add("regards");
		stopWord.add("relatively");
		stopWord.add("respectively");
		stopWord.add("right");
		stopWord.add("s");
		stopWord.add("said");
		stopWord.add("same");
		stopWord.add("saw");
		stopWord.add("say");
		stopWord.add("saying");
		stopWord.add("says");
		stopWord.add("second");
		stopWord.add("secondly");
		stopWord.add("see");
		stopWord.add("seeing");
		// stopWord.add("seem");
		// stopWord.add("seemed");
		// stopWord.add("seeming");
		// stopWord.add("seems");
		stopWord.add("seen");
		stopWord.add("self");
		stopWord.add("selves");
		stopWord.add("sensible");
		stopWord.add("sent");
		// stopWord.add("serious");
		// stopWord.add("seriously");
		stopWord.add("seven");
		stopWord.add("several");
		stopWord.add("shall");
		stopWord.add("she");
		stopWord.add("should");
		stopWord.add("since");
		stopWord.add("six");
		stopWord.add("so");
		stopWord.add("some");
		stopWord.add("somebody");
		stopWord.add("somehow");
		stopWord.add("someone");
		stopWord.add("something");
		stopWord.add("sometime");
		stopWord.add("sometimes");
		stopWord.add("somewhat");
		stopWord.add("somewhere");
		stopWord.add("soon");
		stopWord.add("sorry");
		stopWord.add("specified");
		stopWord.add("specify");
		stopWord.add("specifying");
		stopWord.add("still");
		stopWord.add("sub");
		stopWord.add("such");
		stopWord.add("sup");
		stopWord.add("sure");
		stopWord.add("t");
		stopWord.add("take");
		stopWord.add("taken");
		stopWord.add("tell");
		stopWord.add("tends");
		stopWord.add("th");
		stopWord.add("than");
		stopWord.add("thank");
		stopWord.add("thanks");
		stopWord.add("thanx");
		stopWord.add("that");
		stopWord.add("this");
		stopWord.add("thats");
		stopWord.add("the");
		stopWord.add("their");
		stopWord.add("theirs");
		stopWord.add("them");
		stopWord.add("themselves");
		stopWord.add("then");
		stopWord.add("thence");
		stopWord.add("there");
		stopWord.add("thereafter");
		stopWord.add("thereby");
		stopWord.add("therefore");
		stopWord.add("therein");
		stopWord.add("theres");
		stopWord.add("thereupon");
		stopWord.add("these");
		stopWord.add("they");
		stopWord.add("think");
		stopWord.add("third");
		stopWord.add("this");
		stopWord.add("thorough");
		stopWord.add("thoroughly");
		stopWord.add("those");
		stopWord.add("though");
		stopWord.add("three");
		stopWord.add("through");
		stopWord.add("throughout");
		stopWord.add("thru");
		stopWord.add("thus");
		stopWord.add("to");
		stopWord.add("together");
		stopWord.add("too");
		stopWord.add("took");
		stopWord.add("toward");
		stopWord.add("towards");
		stopWord.add("tried");
		stopWord.add("tries");
		stopWord.add("truly");
		stopWord.add("try");
		stopWord.add("trying");
		stopWord.add("twice");
		stopWord.add("two");
		stopWord.add("u");
		stopWord.add("un");
		stopWord.add("under");
		// stopWord.add("unfortunately");
		// stopWord.add("unless");
		// stopWord.add("unlikely");
		stopWord.add("until");
		stopWord.add("unto");
		stopWord.add("up");
		stopWord.add("upon");
		stopWord.add("us");
		stopWord.add("use");
		stopWord.add("used");
		// stopWord.add("useful");
		stopWord.add("uses");
		stopWord.add("using");
		stopWord.add("usually");
		stopWord.add("uucp");
		stopWord.add("v");
		stopWord.add("value");
		stopWord.add("various");
		stopWord.add("ve"); // added to avoid words like I've,you've etc.
		stopWord.add("very");
		stopWord.add("via");
		stopWord.add("viz");
		stopWord.add("vs");
		stopWord.add("w");
		stopWord.add("want");
		stopWord.add("wants");
		stopWord.add("was");
		// stopWord.add("way");
		stopWord.add("we");
		stopWord.add("welcome");
		// stopWord.add("well");
		stopWord.add("went");
		stopWord.add("were");
		stopWord.add("what");
		// stopWord.add("whatever");
		stopWord.add("when");
		stopWord.add("whence");
		stopWord.add("whenever");
		stopWord.add("where");
		stopWord.add("whereafter");
		stopWord.add("whereas");
		stopWord.add("whereby");
		stopWord.add("wherein");
		stopWord.add("whereupon");
		stopWord.add("wherever");
		stopWord.add("whether");
		stopWord.add("which");
		stopWord.add("while");
		stopWord.add("whither");
		stopWord.add("who");
		stopWord.add("whoever");
		stopWord.add("whole");
		stopWord.add("whom");
		stopWord.add("whose");
		stopWord.add("why");
		stopWord.add("will");
		stopWord.add("willing");
		stopWord.add("wish");
		stopWord.add("with");
		stopWord.add("within");
		stopWord.add("without");
		stopWord.add("wonder");
		stopWord.add("would");
		stopWord.add("would");
		stopWord.add("x");
		stopWord.add("y");
		// stopWord.add("yes");
		stopWord.add("yet");
		stopWord.add("you");
		stopWord.add("your");
		stopWord.add("yours");
		stopWord.add("yourself");
		stopWord.add("yourselves");
		stopWord.add("z");
		stopWord.add("zero");
		// add new
		stopWord.add("i'm");
		stopWord.add("he's");
		stopWord.add("she's");
		stopWord.add("you're");
		stopWord.add("i'll");
		stopWord.add("you'll");
		stopWord.add("she'll");
		stopWord.add("he'll");
		stopWord.add("it's");
		stopWord.add("don't");
		stopWord.add("can't");
		stopWord.add("didn't");
		stopWord.add("i've");
		stopWord.add("that's");
		stopWord.add("there's");
		stopWord.add("isn't");
		stopWord.add("what's");
		stopWord.add("rt");
		stopWord.add("doesn't");
		stopWord.add("w/");
		stopWord.add("w/o");
	}

	public int[] addDocument(List<String> document)
	{
		int size = document.size();
		int[] doc = new int[size];
		String[] info = new String[size];
		int i = 0;
		for (String word : document)
		{
			// System.out.println(word);
			int id = vocabulary.getId(word, true);
			info[i] = word;
			doc[i] = id;
			i++;
		}
		info = countWord(info);

		topicWord.add(count, info);
		documentList.add(doc);
		count++;
		return doc;
	}

	public String addDocument(List<String> document, String folder, String fileName, String originContent)
	{
		int size = document.size();
		int[] doc = new int[size];
		String[] info = new String[size];
		int i = 0;
		for (String word : document)
		{
			// System.out.println(word);
			int id = vocabulary.getId(word, true);
			info[i] = word;
			doc[i] = id;
			i++;
		}

		Map<String, Integer> record = new HashMap<String, Integer>();
		Map<String, Double> frequency = new HashMap<String, Double>();

		/*
		 * for (String word : info) { System.out.print(" " + word); }
		 * System.out.println();
		 */

		info = countWord(info, record);

		/*
		 * for (String word : info) { System.out.print(" " + word); }
		 * System.out.println(); System.out.println(record.size());
		 * System.out.println(info.length);
		 */

		for (int j = 0; j < info.length; j++)
		{
			String word = info[j];
			// System.out.println(word);
			int count = record.get(word);
			frequency.put(word, (count + 0.0) / size);

			// System.out.println(info[j]);
			// System.out.println(frequency.get(word));
		}

		String content = scatter(fileName, folder, frequency, size, originContent);

		return content;
	}

	public String scatter(String fileName, String folder, Map<String, Double> frequency, int size, String originContent)
	{
		String scatterFile = folder + "../scatter_content/" + fileName;

		// String originFile = folder + fileName;
		// Scanner scanner = new Scanner(new File(scatterFile));
		try
		{
			@SuppressWarnings("resource")
			Scanner scanner = new Scanner(new File(scatterFile));
			Map<String, Double> location2frequency = new LinkedHashMap<String, Double>();
			double sumFre = 0;
			int count = 0;
			while (scanner.hasNextLine())
			{
				String line = scanner.nextLine();
				String[] result = line.split(";");
				String city = result[0];
				Double fre = Double.parseDouble(result[1]);
				// sumFre += fre;

				// System.out.println(city);
				// System.out.println(fre);

				location2frequency.put(city, fre);
				if (count < TOP_N_SERVICE)
				{
					sumFre += fre;
				}
				count++;
			}

			double sumFreWord = 0;
			count = 0;
			for (Double value : frequency.values())
			{
				if (count > TOP_N_SERVICE)
				{
					break;
				}

				sumFreWord += value;
				count++;
			}

			double gamma = Math.max(Math.min(OMEGA + sumFre, ETA), DELTA);

			// System.out.println(sumFre);
			// System.out.println(gamma);
			// System.out.println(sumFreWord);

			// int locationCount =
			// System.out.println(size);

			// double addition = (gamma * size * sumFreWord) / (1 - gamma);
			double addition = gamma * size * sumFreWord;
			int addWordNumber = (int) (addition + 0.5);
			// System.out.println("add number:" + addWordNumber);

			Map<String, Integer> addResult = new HashMap<>();

			Iterator iterator = location2frequency.entrySet().iterator();

			int index = 0;
			int number = 0;

			while (iterator.hasNext())
			{
				if (index >= TOP_N_SERVICE)
				{
					break;
				}

				if (number >= addWordNumber)
				{
					break;
				}

				Entry<?, ?> entry = (Entry<?, ?>) iterator.next();

				String word = (String) entry.getKey();
				// System.out.println(entry.getValue());
				int add = (int) (addWordNumber * (double) (entry.getValue()) + 1);
				addResult.put(word, add);

				// System.out.println(word + " : " + add);
				index++;
				number += add;
			}

			// Todo: fill the frequency word to the origin content.
			// System.out.println(originContent);
			String[] temp = originContent.split(" ");
			// ArrayList<String> list = new ArrayList<> (new String[] { "aaa",
			// "bbb" });

			ArrayList<String> list = new ArrayList<String>();
			for (int i = 0; i < temp.length; i++)
			{
				list.add(temp[i]);
				// System.out.println(list.get(i));
			}
			// String[] out = new String[temp.length + addResult.size()];

			Iterator<?> it = addResult.entrySet().iterator();
			String outContent = "";

			Random random = new Random();

			while (it.hasNext())
			{
				Entry<?, ?> entry = (Entry<?, ?>) it.next();

				String word = (String) entry.getKey();
				int value = (int) entry.getValue();

				// System.out.println("add the word: " + word + " " + value);

				for (int j = 0; j < value; j++)
				{
					int choose = random.nextInt(list.size());
					list.add(choose, word);
					// System.out.println(list.get(choose));
				}
			}

			for (int i = 0; i < list.size(); i++)
			{
				outContent = outContent + list.get(i) + " ";
			}

			// System.out.println(outContent);

			outContent = outContent.trim();
			writeScatteredFile(folder, fileName, outContent);
			return outContent;

		} catch (FileNotFoundException e)
		{
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.exit(-2);
		}

		return null;
	}

	private void writeScatteredFile(String folder, String fileName, String content)
	{
		// TODO Auto-generated method stub
		String outFileName = folder + "../scattered_content/" + fileName;

		File file = new File(outFileName);
		// System.out.println(file.getAbsolutePath());
		if (!file.exists())
		{
			try
			{
				file.createNewFile();
			} catch (IOException e)
			{
				// TODO Auto-generated catch block
				e.printStackTrace();
				System.exit(-3);
			}
		}

		FileWriter fw;
		try
		{
			fw = new FileWriter(file);
			fw.write(content);
			fw.close();
		} catch (IOException e)
		{
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.exit(-4);
		}
	}

	public static boolean containStopWord(String word)
	{
		if (word != null && stopWord.contains(word))
		{
			return true;
		}
		return false;
	}

	public static boolean containStr(String[] array, String aim)
	{
		int hash = aim.hashCode();
		for (int i = 0; i < array.length; i++)
		{
			if (array[i] != null && array[i].hashCode() == hash)
			{
				return true;
			}
		}
		return false;
	}

	public static String[] countWord(String[] words)
	{
		Map<String, Integer> record = new HashMap<>();
		for (int i = 0; i < words.length; i++)
		{
			String word = words[i];
			if (record.containsKey(word))
			{
				Integer value = record.get(word);
				record.put(word, value + 1);
			} else
			{
				record.put(word, new Integer(1));
			}
		}

		String[] result = new String[RANK];

		int minValue = -1, minCount = 0, position = 0;

		for (int i = 0; i < words.length; i++)
		{
			String word = words[i];
			Integer value = record.get(word);

			if (containStr(result, word))
			{
				continue;
			}

			if (position < RANK)
			{
				if (minValue == -1)
				{
					result[position] = word;
					minCount = 0;
					minValue = value.intValue();
				} else if (value.intValue() > minValue)
				{
					result[position] = word;
				} else
				{
					minValue = value.intValue();
					minCount = position;
					result[position] = word;
				}
				position++;
			} else if (value.intValue() > minValue)
			{
				result[minCount] = word;
				Integer min = record.get(result[0]);
				minCount = 0;

				for (int j = 1; j < RANK; j++)
				{
					Integer compare = record.get(result[j]);

					if (min.intValue() > compare.intValue())
					{
						min = compare;
						minCount = j;
					}
				}
			}
		}
		return result;
	}

	public static String[] countWord(String[] words, Map<String, Integer> record)
	{
		// word2map.clear();
		if (record != null)
		{
			record.clear();
		} else
		{
			record = new HashMap<>();
		}

		// System.out.println(words.length);

		for (int i = 0; i < words.length; i++)
		{
			String word = words[i];
			if (record.containsKey(word))
			{
				Integer value = record.get(word);
				record.put(word, value + 1);
			} else
			{
				record.put(word, new Integer(1));
			}
		}

		String[] result = new String[RANK];

		int minValue = -1, minCount = 0, position = 0;

		for (int i = 0; i < words.length; i++)
		{
			String word = words[i];
			Integer value = record.get(word);

			if (containStr(result, word))
			{
				continue;
			}

			if (position < RANK)
			{
				if (minValue == -1)
				{
					result[position] = word;
					minCount = 0;
					minValue = value.intValue();
				} else if (value.intValue() > minValue)
				{
					result[position] = word;
				} else
				{
					minValue = value.intValue();
					minCount = position;
					result[position] = word;
				}
				position++;
			} else if (value.intValue() > minValue)
			{
				result[minCount] = word;
				Integer min = record.get(result[0]);
				minCount = 0;

				for (int j = 1; j < RANK; j++)
				{
					Integer compare = record.get(result[j]);

					if (min.intValue() > compare.intValue())
					{
						min = compare;
						minCount = j;
					}
				}
			}
		}

		// System.out.println("position : " + position);
		String[] newResult = new String[position];

		for (int i = 0; i < position; i++)
		{
			newResult[i] = result[i];
		}
		return newResult;
	}

	public int[][] toArray()
	{
		return documentList.toArray(new int[0][]);
	}

	public int getVocabularySize()
	{
		return vocabulary.size();
	}

	@Override
	public String toString()
	{
		final StringBuilder sb = new StringBuilder();
		for (int[] doc : documentList)
		{
			sb.append(Arrays.toString(doc)).append("\n");
		}
		sb.append(vocabulary);
		return sb.toString();
	}

	/**
	 * Load documents from disk
	 *
	 * @param folderPath
	 *            is a folder, which contains text documents.
	 * @return a corpus
	 * @throws IOException
	 */
	public static Corpus load(String folderPath) throws IOException
	{
		Corpus corpus = new Corpus();
		File folder = new File(folderPath);
		
		File scatterFolder = new File(folderPath + "../scatter_content/");
		scatterList = new ArrayList<String>();
		
		for(File file : scatterFolder.listFiles())
		{
			scatterList.add(file.getName());
		}
		
		int word_length = 0;
		
		for (File file : folder.listFiles())
		{
			String fileName = file.getName();
			// System.out.println(fileName);
			if(scatterList.indexOf(fileName) == -1)
			{
				System.err.println("File :" + fileName + " not found.");
				continue;
			}
			// Corpus.documentName.add(fileName);
			BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
			String line;
			List<String> wordList = new LinkedList<String>();

			String reg = "[^a-zA-Z]";
			String originContent = "";

			while ((line = br.readLine()) != null)
			{
				originContent += line;
				// char temp = '　';
				// line = line.replace(temp, ' ');
				line = line.replaceAll(reg, " ");
				line = line.toLowerCase();
				
				String[] words = line.split(" |\t|\n|	　|	");
				
				for (String word : words)
				{
				
					/*
					word = word.replace(temp, ' ');
					word = word.replaceAll(reg, " ");
					word = word.toLowerCase();
					*/
					
					/*
					if(word.hashCode() == "adpromo".hashCode())
					{
						System.out.println("find the adpromo");
						System.out.println(word);
						System.out.println(fileName);
						System.exit(-4);
					}
					*/
					
					// System.out.println(Inflector.getInstance().pluralize("books"));

					/*
					 * word = word.replaceAll("\\s*", ""); word = word.trim();
					 * word = word.replaceAll("\\s", "");
					 */

					// System.out.println(word);

					if (containStopWord(word))
					{
						continue;
					}

					word = Inflector.getInstance().singularize(word);
					
					/*
					 * for(int i = 0; i < dictionary.length; i++) {
					 * if(word.hashCode() == dictionary[i].hashCode()) {
					 * System.out.println(word); continue; } }
					 */
					
					if (word.length() < 2)
					{
						continue;
					}
					
					// System.out.println("add the word " + word);
					wordList.add(word);
				}
			}
			br.close();

			// String[] temp = originContent.split(" ");
			/*
			if (wordList.size() < RANK)
			{
				moveToFailed(folderPath, fileName);
				continue;
			}
			*/
			
			word_length += wordList.size();
			
			line = corpus.addDocument(wordList, folderPath, fileName, originContent);
			
			Corpus.documentName.add(fileName);			

			// System.out.println(line);
			wordList.clear();

			{
				// originContent += line;
				// char temp = '　';

				// line = line.replace(temp, ' ');
				
				line = line.replaceAll(reg, " ");
				line = line.toLowerCase();
				
				String[] words = line.split(" |\t|\n|	　|	");
				for (String word : words)
				{

					// System.out.println(Inflector.getInstance().pluralize("books"));

					/*
					 * word = word.replaceAll("\\s*", ""); word = word.trim();
					 * word = word.replaceAll("\\s", "");
					 */

					// System.out.println(word);

					if (containStopWord(word))
					{
						continue;
					}

					// word = Inflector.getInstance().singularize(word);

					/*
					 * for(int i = 0; i < dictionary.length; i++) {
					 * if(word.hashCode() == dictionary[i].hashCode()) {
					 * System.out.println(word); continue; } }
					 */

					if (word.length() < 2)
					{
						continue;
					}

					// System.out.println("add the word " + word);
					wordList.add(word);
				}
			}

			
			corpus.addDocument(wordList);
			// System.out.println(wordList.toString());
			// System.exit(0);
		}
		
		String vocabularyFile = "./vocabular.txt";
		
		// double avg = (double)(word_length) / documentList.size();
		// System.out.println("avg length:" + avg);
		
		
		// System.out.println(corpus.vocabulary);
		// System.exit(-7);
		
		saveTheWordList(folderPath, corpus.vocabulary.word2idMap, vocabularyFile);
		
		if (corpus.getVocabularySize() == 0)
		{
			return null;
		}

		return corpus;
	}

	private static void saveTheWordList(String path, Map<String, Integer> word2idMap, String vocabularyFile)
	{
		// TODO Auto-generated method stub
		String outFileName = path + "../" + vocabularyFile;

		File file = new File(outFileName);
		// System.out.println(file.getAbsolutePath());
		if (!file.exists())
		{
			try
			{
				file.createNewFile();
			} catch (IOException e)
			{
				// TODO Auto-generated catch block
				e.printStackTrace();
				System.exit(-3);
			}
		}

		FileWriter fw;
		try
		{
			fw = new FileWriter(file);
			
			String content = "";
			
			for(String key: word2idMap.keySet())
			{
				content += key + "\n";
			}
			
			fw.write(content);
			fw.close();
		} catch (IOException e)
		{
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.exit(-4);
		}
	}

	private static void moveToFailed(String folderPath, String fileName)
	{
		// TODO Auto-generated method stub
		File oldFile = new File(folderPath + fileName);
		File newFile = new File(folderPath + "../failed/" + fileName);

		if (oldFile.renameTo(newFile))
		{
			System.out.println("Origin File: " + fileName + " moved.");
		} else
		{
			System.err.println("Origin File: " + fileName + " moved filed!");
			System.exit(-6);
		}
	}

	public Vocabulary getVocabulary()
	{
		return vocabulary;
	}

	public int[][] getDocument()
	{
		return toArray();
	}

	public static int[] loadDocument(String path, Vocabulary vocabulary) throws IOException
	{
		BufferedReader br = new BufferedReader(new FileReader(path));
		String line;
		List<Integer> wordList = new LinkedList<Integer>();
		while ((line = br.readLine()) != null)
		{
			String[] words = line.split(" ");
			for (String word : words)
			{
				if (word.trim().length() < 2)
					continue;
				Integer id = vocabulary.getId(word);
				if (id != null)
					wordList.add(id);
			}
		}
		br.close();
		int[] result = new int[wordList.size()];
		int i = 0;
		for (Integer integer : wordList)
		{
			result[i++] = integer;
		}
		return result;
	}
}
