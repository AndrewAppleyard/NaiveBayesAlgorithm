import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map.Entry;
import java.util.Set;

public class UASimpleClassifier {

	HashMap<String, ArrayList<HashMap<String, Double>>> probabilities;
	HashMap<String, ArrayList<HashMap<String, Integer>>> data;

	public UASimpleClassifier() {
		probabilities = new HashMap<>();
		data = new HashMap<>();
	}

	public void train(String filename) {
		int records = 0;
		try {
			String line;
			BufferedReader br = new BufferedReader(new FileReader(filename));
			br.readLine();
			while ((line = br.readLine()) != null) {
				records++;
				String[] token = line.split(",");
				ArrayList<HashMap<String, Integer>> features;
				if (data.get(token[5]) == null) {
					features = new ArrayList<>(5);
					for (int i = 0; i < token.length; i++) {
						HashMap<String, Integer> feature = new HashMap<>();
						features.add(i, feature);
					}
					data.put(token[5], features);
				} else {
					features = data.get(token[5]);
				}

				for (int i = 0; i < token.length; i++) {
					HashMap<String, Integer> hm = features.get(i);
					int count = hm.getOrDefault(token[i], 0);
					count++;
					hm.put(token[i], count);
				}
			}
		} catch (IOException e) {
			e.printStackTrace();
		}

		Set<String> set = data.keySet();
		for (String className : set) {

			ArrayList<HashMap<String, Integer>> list = data.get(className);
			ArrayList<HashMap<String, Double>> features = new ArrayList<HashMap<String, Double>>(list.size());

			for (int i = 0; i < list.size(); i++) {
				int featureSum = 0;
				double continousSum = 0;
				HashMap<String, Integer> count = list.get(i);
				HashMap<String, Double> prob = new HashMap<String, Double>();
				Collection<Entry<String, Integer>> entries = count.entrySet();

				for (Entry<String, Integer> entry : entries) {
					if (i == 3 || i == 4) {
						continousSum += (Double.parseDouble(entry.getKey()) * (double) entry.getValue());
					}
					featureSum += entry.getValue();
				}
				// continuous
				if (i == 3 || i == 4) {
					double mu = continousSum / (double) featureSum;
					double std;
					double sum = 0;
					for (Entry<String, Integer> entry : entries) {
						for (int j = 0; j < entry.getValue(); j++) {
							sum += Math.pow((Double.parseDouble(entry.getKey()) - mu), 2);
						}
					}
					std = Math.sqrt(sum / (double) featureSum);
					prob.put("mu", mu);
					prob.put("std", std);
				} else {
					// discrete
					for (Entry<String, Integer> entry : entries) {
						double prob1;
						if (className.equals(entry.getKey()) && i == list.size() - 1) {
							prob1 = (double) featureSum / (double) records;
						} else {
							prob1 = (double) entry.getValue() / featureSum;
						}
						prob.put(entry.getKey(), prob1);
					}
				}

				features.add(prob);
			}

			probabilities.put(className, features);
		}

		System.out.println("Training Phase:\t  " + filename);
		System.out.println("--------------------------------------------------------------------");
		System.out.println("\t=> Number of Entries (n):\t\t" + records);
		System.out.println("\t=> Number of Features (p):\t\t" + (data.get("0").size() - 1)); // class
		System.out.println("\t=> Number of Distinct Classes (y):\t" + data.keySet().size());
		System.out.println();
	}

	public void test(String filename) throws IOException {
		System.out.println("Testing Phase:\t  " + filename);
		System.out.println("--------------------------------------------------------------------");
		System.out.printf("%-8s %-5s %-5s %-9s    %-8s %-8s %-8s %-4s   %-8s\n", "F1", "F2", "F3", "F4", "F5", "CLASS", "PREDICT", "PROB", "RESULT");
		System.out.printf("%-8s %-5s %-5s %-9s    %-8s %-8s %-8s %-4s  %-8s\n", "---", "---", "---", "-------", "-------", "-------", "-------", "-----", "---------");
		String line = "";
		BufferedReader br = new BufferedReader(new FileReader(filename));
		int correct = 0;
		int records = 0;
		int count = 500;
		line = br.readLine();
		while ((line = br.readLine()) != null) {
			records++;
			String result = "INCORRECT";
			String[] tokens = line.split(",");
			double classProb = classifyProb(tokens[0], tokens[1], tokens[2], Double.parseDouble(tokens[3]), Double.parseDouble(tokens[4]));
			String predictClass = classify(tokens[0], tokens[1], tokens[2], Double.parseDouble(tokens[3]), Double.parseDouble(tokens[4]));

			if (predictClass.equals("0")) {
				if (tokens[5].equals("0")) {
					result = "CORRECT";
				}
			} else {
				if (tokens[5].equals("1")) {
					result = "CORRECT";
				}
			}

			if (count > 0) {
				System.out.printf("%-8s %-5s %-5s %-9.2f    %-8s class%-3s class%-3s %-4.1f%%  %-8s\n", tokens[0],tokens[1], tokens[2], Double.parseDouble(tokens[3]), Double.parseDouble(tokens[4]), tokens[5], predictClass, classProb * 100, result);
				count--;
			}
			if (result.equals("CORRECT")) {
				correct++;
			}
		}
		System.out.println();
		System.out.printf("\tTotal Accuracy:\t   %5d correct /%5d total  =  %5.2f%% Accuracy\n", correct, records, ((double) correct / records) * 100);
		System.out.println();
		System.out.println("\t=>  Number of Entries (n):\t   " + records);
		System.out.println();

	}

	public String classify(String f1, String f2, String f3, double f4, double f5) {
		double docSent = 0;
		double class0 = 0;
		double class1 = 0;
		Set<String> keys = probabilities.keySet();

		for (String key : keys) {
			ArrayList<HashMap<String, Double>> list = probabilities.get(key);
			double prob;
			// naive bayes
			prob = list.get(0).get(f1) * list.get(1).get(f2) * list.get(2).get(f3) * pdf(f4, list.get(3).get("mu"), list.get(3).get("std")) * pdf(f5, list.get(4).get("mu"), list.get(4).get("std")) * list.get(5).get(key);
			if (key.equals("0")) {
				class0 = prob;
			} else {
				class1 = prob;
			}
			docSent += prob;
		}

		class0 = class0 / docSent;
		class1 = class1 / docSent;
		if (class0 > class1) {
			return "0";
		} else {
			return "1";
		}
	}

	public double classifyProb(String f1, String f2, String f3, double f4, double f5) {
		double docSent = 0;
		double class0 = 0;
		double class1 = 0;
		Set<String> keys = probabilities.keySet();

		for (String key : keys) {
			ArrayList<HashMap<String, Double>> list = probabilities.get(key);
			double prob;
			// naive bayes
			prob = list.get(0).get(f1) * list.get(1).get(f2) * list.get(2).get(f3) * pdf(f4, list.get(3).get("mu"), list.get(3).get("std")) * pdf(f5, list.get(4).get("mu"), list.get(4).get("std")) * list.get(5).get(key);
			if (key.equals("0")) {
				class0 = prob;
			} else {
				class1 = prob;
			}
			docSent += prob;
		}

		// n/d
		class0 = class0 / docSent;
		class1 = class1 / docSent;
		if (class0 > class1) {
			return class0;
		} else {
			return class1;
		}
	}

	public void predict(String filename) throws NumberFormatException, IOException {
		System.out.println("Prediction Phase:\t  " + filename);
		System.out.println("--------------------------------------------------------------------");
		System.out.printf(" %-8s %-5s %-5s %-9s    %-8s  %-8s %-6s \n", "F1", "F2", "F3", "F4", "F5", "PREDICT", "PROB");
		System.out.printf(" %-8s %-5s %-5s %-9s    %-8s  %-8s %-6s \n", "-------", "---", "---", "---------", "------", "-------", "-----");
		String line = "";
		BufferedReader br = new BufferedReader(new FileReader(filename));
		int records = 0;
		line = br.readLine();
		while ((line = br.readLine()) != null) {
			String[] tokens = line.split(",");
			String predictClass = classify(tokens[0], tokens[1], tokens[2], Double.parseDouble(tokens[3]), Double.parseDouble(tokens[4]));
			double cprob = classifyProb(tokens[0], tokens[1], tokens[2], Double.parseDouble(tokens[3]), Double.parseDouble(tokens[4]));
			System.out.printf(" %-8s %-5s %-5s %-9.2f    %-8s  class%-3s %-2.1f%% \n", tokens[0], tokens[1], tokens[2], Double.parseDouble(tokens[3]), Double.parseDouble(tokens[4]), predictClass, cprob * (double) 100);
			records++;
		}

		System.out.println();
		System.out.println("\t=> Number of Entries (n):\t\t" + records);
	}

	public static void main(String[] args) {

		System.out.println("************************************************************");
		System.out.println("Problem Set:\t Problem Set 3:   Naive Bayes Algorithm");
		System.out.println("Name:\t\t Andrew Appleyard");
		System.out.println("Syntax:\t\t java UASimpleClassifier arg1 arg2 arg3");
		System.out.println("************************************************************");
		System.out.println();

		if (args.length != 3) {
			System.out.println("java UASimpleClassifier train.txt test.txt predict.txt");
			return;
		} else {
			System.out.println("Follow this format: java UASimpleClassifier train.txt test.txt predict.txt");
		}
		UASimpleClassifier x = new UASimpleClassifier();
		x.train(args[0]);
		try {
			x.test(args[1]);
		} catch (IOException e1) {
			e1.printStackTrace();
		}
		try {
			x.predict(args[2]);
		} catch (NumberFormatException e1) {
			e1.printStackTrace();
		} catch (IOException e1) {
			e1.printStackTrace();
		}

		int totalFeatures = 5;

		String train = args[0];
		// String test = args[1];
		// String predict = args[2];

		HashMap<String, Integer> geographyClass0 = new HashMap<>();
		HashMap<String, Integer> geographyClass1 = new HashMap<>();

		try (BufferedReader br = new BufferedReader(new FileReader(train))) {
			String line;
			br.readLine();

			while ((line = br.readLine()) != null) {
				String[] values = line.split(",");
				if (values.length == 6) {
					totalFeatures = values.length;
					String geographyCol = values[0];
					int exited = Integer.parseInt(values[5]);

					if (exited == 0) {
						if (geographyClass0.containsKey(geographyCol)) {
							int currentCount = geographyClass0.get(geographyCol);
							geographyClass0.put(geographyCol, currentCount + 1);
							// System.out.printf("Updating %s -> %d to %s -> %d%n", geographyCol,
							// currentCount, geographyCol, currentCount + 1);
						} else {
							geographyClass0.put(geographyCol, 1);
							// System.out.printf("*** Inserting <%s,%d>%n", geographyCol, 1);
						}
					} else {
						if (geographyClass1.containsKey(geographyCol)) {
							int currentCount = geographyClass1.get(geographyCol);
							geographyClass1.put(geographyCol, currentCount + 1);
							// System.out.printf("Updating %s -> %d to %s -> %d%n", geographyCol,
							// currentCount, geographyCol, currentCount + 1);
						} else {
							geographyClass1.put(geographyCol, 1);
							// System.out.printf("*** Inserting <%s,%d>%n", geographyCol, 1);
						}
					}
				}
			}
		} catch (IOException e) {
			e.printStackTrace();
		}

		HashMap<String, Integer> activeMemberClass0 = new HashMap<>();
		HashMap<String, Integer> activeMemberClass1 = new HashMap<>();

		try (BufferedReader br = new BufferedReader(new FileReader(train))) {
			String line;
			br.readLine();

			while ((line = br.readLine()) != null) {
				String[] values = line.split(",");
				if (values.length == 6) {
					totalFeatures = values.length;
					String activeMemberCol = values[1];
					int exited = Integer.parseInt(values[5]);

					if (exited == 0) {
						if (activeMemberClass0.containsKey(activeMemberCol)) {
							int currentCount = activeMemberClass0.get(activeMemberCol);
							activeMemberClass0.put(activeMemberCol, currentCount + 1);
							// System.out.printf("Updating %s -> %d to %s -> %d%n", activeMemberCol,
							// currentCount, activeMemberCol, currentCount + 1);
						} else {
							activeMemberClass0.put(activeMemberCol, 1);
							// System.out.printf("*** Inserting <%s,%d>%n", activeMemberCol, 1);
						}
					} else {
						if (activeMemberClass1.containsKey(activeMemberCol)) {
							int currentCount = activeMemberClass1.get(activeMemberCol);
							activeMemberClass1.put(activeMemberCol, currentCount + 1);
							// System.out.printf("Updating %s -> %d to %s -> %d%n", activeMemberCol,
							// currentCount, activeMemberCol, currentCount + 1);
						} else {
							activeMemberClass1.put(activeMemberCol, 1);
							// System.out.printf("*** Inserting <%s,%d>%n", activeMemberCol, 1);
						}
					}
				}
			}
		} catch (IOException e) {
			e.printStackTrace();
		}

		HashMap<String, Integer> hasCrCardClass0 = new HashMap<>();
		HashMap<String, Integer> hasCrCardClass1 = new HashMap<>();

		try (BufferedReader br = new BufferedReader(new FileReader(train))) {
			String line;
			br.readLine();

			while ((line = br.readLine()) != null) {
				String[] values = line.split(",");
				if (values.length == 6) {
					totalFeatures = values.length;
					String hasCrCardCol = values[2];
					int exited = Integer.parseInt(values[5]);

					if (exited == 0) {
						if (hasCrCardClass0.containsKey(hasCrCardCol)) {
							int currentCount = hasCrCardClass0.get(hasCrCardCol);
							hasCrCardClass0.put(hasCrCardCol, currentCount + 1);
							// System.out.printf("Updating %s -> %d to %s -> %d%n", hasCrCardCol,
							// currentCount, hasCrCardCol, currentCount + 1);
						} else {
							hasCrCardClass0.put(hasCrCardCol, 1);
							// System.out.printf("*** Inserting <%s,%d>%n", hasCrCardCol, 1);
						}
					} else {
						if (hasCrCardClass1.containsKey(hasCrCardCol)) {
							int currentCount = hasCrCardClass1.get(hasCrCardCol);
							hasCrCardClass1.put(hasCrCardCol, currentCount + 1);
							// System.out.printf("Updating %s -> %d to %s -> %d%n", hasCrCardCol,
							// currentCount, hasCrCardCol, currentCount + 1);
						} else {
							hasCrCardClass1.put(hasCrCardCol, 1);
							// System.out.printf("*** Inserting <%s,%d>%n", hasCrCardCol, 1);
						}
					}
				}
			}
		} catch (IOException e) {
			e.printStackTrace();
		}

		HashMap<Double, Double> balanceClass0 = new HashMap<>();
		HashMap<Double, Double> balanceClass1 = new HashMap<>();
		double balanceClass0Count = 0;
		double balanceClass1Count = 0;

		try (BufferedReader br = new BufferedReader(new FileReader(train))) {
			String line;
			br.readLine();

			while ((line = br.readLine()) != null) {
				String[] values = line.split(",");
				if (values.length == 6) {
					double balanceCol = Double.parseDouble(values[3]);
					int exited = Integer.parseInt(values[5]);

					if (exited == 0) {
						balanceClass0Count++;
						if (balanceClass0.containsKey(balanceCol)) {
							double currentCount = balanceClass0.get(balanceCol);
							balanceClass0.put(balanceCol, currentCount + 1);
							// System.out.printf("Updating %f -> %f to %f -> %f%n", balanceCol,
							// balanceClass0Count, balanceCol, balanceClass0Count + 1);
						} else {
							balanceClass0.put(balanceCol, 1.0);
							// System.out.printf("*** Inserting <%f,%d>%n", balanceCol, 1);
						}
					} else {
						balanceClass1Count++;
						if (balanceClass1.containsKey(balanceCol)) {
							double currentCount = balanceClass1.get(balanceCol);
							balanceClass1.put(balanceCol, currentCount + 1);
							// System.out.printf("Updating %f -> %f to %f -> %f%n", balanceCol,
							// balanceClass1Count, balanceCol, balanceClass1Count + 1);
						} else {
							balanceClass1.put(balanceCol, 1.0);
							// System.out.printf("*** Inserting <%f,%d>%n", balanceCol, 1);
						}
					}
				}
			}
		} catch (IOException e) {
			e.printStackTrace();
		}

		HashMap<Double, Double> creditScoreClass0 = new HashMap<>();
		HashMap<Double, Double> creditScoreClass1 = new HashMap<>();
		double creditScoreClass0Count = 0;
		double creditScoreClass1Count = 0;

		try (BufferedReader br = new BufferedReader(new FileReader(train))) {
			String line;
			br.readLine();

			while ((line = br.readLine()) != null) {
				String[] values = line.split(",");
				if (values.length == 6) {
					double creditScoreCol = Double.parseDouble(values[4]);
					int exited = Integer.parseInt(values[5]);

					if (exited == 0) {
						creditScoreClass0Count++;
						if (creditScoreClass0.containsKey(creditScoreCol)) {
							double currentCount = creditScoreClass0.get(creditScoreCol);
							creditScoreClass0.put(creditScoreCol, currentCount + 1);
							// System.out.printf("Updating %f -> %f to %f -> %f%n", creditScoreCol,
							// creditScoreClass0Count, creditScoreCol, creditScoreClass0Count + 1);
						} else {
							creditScoreClass0.put(creditScoreCol, 1.0);
							// System.out.printf("*** Inserting <%f,%d>%n", creditScoreCol, 1);
						}
					} else {
						creditScoreClass1Count++;
						if (creditScoreClass1.containsKey(creditScoreCol)) {
							double currentCount = creditScoreClass1.get(creditScoreCol);
							creditScoreClass1.put(creditScoreCol, currentCount + 1);
							// System.out.printf("Updating %f -> %f to %f -> %f%n", creditScoreCol,
							// creditScoreClass1Count, creditScoreCol, creditScoreClass1Count + 1);
						} else {
							creditScoreClass1.put(creditScoreCol, 1.0);
							// System.out.printf("*** Inserting <%f,%d>%n", creditScoreCol, 1);
						}
					}
				}
			}
		} catch (IOException e) {
			e.printStackTrace();
		}

		// discreteProbPrintMethod(geographyClass0, geographyClass1, "geography");
		// discreteProbPrintMethod(activeMemberClass0, activeMemberClass1,
		// "activeMember");
		// discreteProbPrintMethod(hasCrCardClass0, hasCrCardClass1, "hasCrCard");
		// continuousProbPrintMethod(balanceClass0, balanceClass1, balanceClass0Count,
		// balanceClass1Count, "balance");
		// continuousProbPrintMethod(creditScoreClass0, creditScoreClass1,
		// creditScoreClass0Count, creditScoreClass1Count,"creditScore");

	}

	private static void discreteProbPrintMethod(HashMap<String, Integer> class0, HashMap<String, Integer> class1,
			String name) {
		int totalEntries = 0;
		for (int count : class0.values()) {
			totalEntries += count;
		}
		for (int count : class1.values()) {
			totalEntries += count;
		}

		// Display counts for name when Exited is 0
		System.out.println("Counts for " + name + " when exited is 0:");
		for (String key : class0.keySet()) {
			System.out.println(key + ": " + class0.get(key));
		}

		// Display counts for - when Exited is 1
		System.out.println("\nCounts for " + name + " when Exited is 1:");
		for (String key : class1.keySet()) {
			System.out.println(key + ": " + class1.get(key));
		}

		System.out.println();
		System.out.println("Printing out all values in hash table:");
		System.out.println();

		System.out.printf("%3s       %s      %s %n", "Key", "Value", "Prob");
		System.out.printf("%3s       %s      %s %n", "---", "-----", "----");

		for (String key : class0.keySet()) {
			double prob = (double) class0.get(key) / totalEntries;
			System.out.printf("%3s  -->  %3d        %-1.3f %n", key, class0.get(key), prob);
		}

		for (String key : class1.keySet()) {
			double prob = (double) class1.get(key) / totalEntries;
			System.out.printf("%3s  -->  %3d        %-1.3f %n", key, class1.get(key), prob);
		}
		System.out.println("\n\n");
	}

	private static void continuousProbPrintMethod(HashMap<Double, Double> class0, HashMap<Double, Double> class1,
			double class0Count, double class1Count, String name) {
		double totalFeatures = class0Count + class1Count;

		// Display counts for name when Exited is 0
		// System.out.println("Counts for " + name + " when exited is 0:");
		// for (Double key : class0.keySet()) {
		// System.out.println(key + ": " + class0.get(key));
		// }

		// Display counts for - when Exited is 1
		// System.out.println("\nCounts for " + name + " when Exited is 1:");
		// for (Double key : class1.keySet()) {
		// System.out.println(key + ": " + class1.get(key));
		// }

		// System.out.println("Printing out all values in hash table");
		// System.out.println();
		// System.out.printf("%10s %s %s %n", "Key", "Value", "Prob");
		// System.out.printf("%10s %s %s %n", "----------", "----------", "----------");

		for (Double key : class0.keySet()) {
			double count = class0.get(key);
			double prob = pdf(key, mean(class0), sd(class0)) * class0Count / totalFeatures;
			// System.out.printf("%3s --> %3d %-1.9f %n", key, (int) count, prob);
		}
		System.out.println("\n\n\n");
		for (Double key : class1.keySet()) {
			double count = class1.get(key);
			double prob = pdf(key, mean(class1), sd(class1)) * class1Count / totalFeatures;
			// System.out.printf("%3s --> %3d %-1.9f %n", key, (int) count, prob);
		}
		System.out.println("\n\n");
	}

	private static double pdf(double x, double mu, double sigma) {
		double z = (x - mu) / sigma;
		return Math.exp(-z * z / 2) / Math.sqrt(2 * Math.PI * sigma * sigma);
	}

	private static double mean(HashMap<Double, Double> data) {
		double sum = 0;
		for (double key : data.keySet()) {
			sum += key * data.get(key);
		}
		return sum / data.size();
	}

	private static double sd(HashMap<Double, Double> data) {
		double mean = mean(data);
		double sum = 0;
		for (double key : data.keySet()) {
			sum += Math.pow(key - mean, 2) * data.get(key);
		}
		return Math.sqrt(sum / data.size());
	}

}
