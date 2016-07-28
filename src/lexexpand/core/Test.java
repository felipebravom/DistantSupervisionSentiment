package lexexpand.core;


import java.io.BufferedReader;
import java.io.FileReader;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.LibLINEAR;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.DistantSupervisionFilter;
import weka.filters.unsupervised.attribute.EmoticonDistantSupervision;
import weka.filters.unsupervised.attribute.PTCM;
import weka.filters.unsupervised.attribute.RemoveType;
import weka.filters.unsupervised.attribute.ASA;


public class Test {



	public Test(){

	}




	public Instances mapTargetData(String input, DistantSupervisionFilter distantFilt) throws Exception{
		BufferedReader readerTest = new BufferedReader(
				new FileReader(input));

		Instances corpus = new Instances(readerTest);


		Filter.useFilter(corpus, distantFilt);

		//Instances targetData=distantFilt.mapTargetInstance(corpus);

		Instances targetData=Filter.useFilter(corpus, distantFilt);

		//	targetData.setClassIndex(targetData.numAttributes()-1);


		targetData.setClassIndex(targetData.numAttributes()-1);

		return targetData;

	}




	// Use a classifier trained from word-label for classifying tweets 

	public void evaluateDataSet(Instances trainData,String path, DistantSupervisionFilter distantFilt) throws Exception{
		Instances targetData=this.mapTargetData(path,distantFilt);


		LibLINEAR ll=new LibLINEAR();
		ll.setOptions(Utils.splitOptions("-S 7 -C 1.0 -E 0.01 -B 1.0 -P"));

		RemoveType rm=new RemoveType();

		rm.setOptions(Utils.splitOptions("-T String"));

		FilteredClassifier fc = new FilteredClassifier();
		fc.setFilter(rm);
		fc.setClassifier(ll);
		fc.buildClassifier(trainData);




		// 
		Evaluation targetEval = new weka.classifiers.Evaluation(trainData);


		targetEval.evaluateModel(fc, targetData);

		System.out.println("Results on "+path);
		System.out.println("kappa,"+targetEval.kappa());
		System.out.println("AvgF1,"+(targetEval.fMeasure(0)+targetEval.fMeasure(1))/2);
		System.out.println("AUC,"+targetEval.weightedAreaUnderROC());


	}









	public void processDistFilt(Instances sourceData,DistantSupervisionFilter distFiltEx) throws Exception{



		distFiltEx.setInputFormat(sourceData);
		//		distFiltEx.setTweetsPerCentroid(10);			
		//		System.out.println("DistFilt Tweets per Centroid "+distFiltEx.getTweetsPerCentroid());

		Instances trainDistData=Filter.useFilter(sourceData, distFiltEx);
		trainDistData.setClassIndex(trainDistData.numAttributes()-1);
		//
		//
		//
		System.out.println(distFiltEx.getUsefulInfo());
		System.out.println("Dist Model Attributes,"+ trainDistData.numAttributes());
		System.out.println("Dist ModelInstances,"+ trainDistData.numInstances());
		this.evaluateDataSet(trainDistData, "example/6HumanPosNeg.arff", distFiltEx);






	}



	// Take an input collection of tweets partionate it 10 versions, create different datasets and use them for evaluation


	// 	edimEx.arff	experiment/   1

	// 	edimEx.arff	experiment/   1 10 10000
	static public void main(String args[]) throws Exception{

		// Input String in args[0]
		String inputFile=args[0];

		Test wlf=new Test();

		

		DistantSupervisionFilter	emoFilter=new EmoticonDistantSupervision();
		emoFilter.setOptions(Utils.splitOptions("-M 1 -W -C -I 1 -P WORD- -Q CLUST- -L -H resources/50mpaths2.txt -T resources/stopwords.txt -O"));

		

		DistantSupervisionFilter asaFilter=new ASA();		
		// Non mutually exclusive false			
		asaFilter.setOptions(Utils.splitOptions("-M 1 -W -C -I 1 -P WORD- -Q CLUST- -L -J lexicons/AFINN-posneg.txt -H resources/50mpaths2.txt -T resources/stopwords.txt -O -A 10 -B 1.0"));



		DistantSupervisionFilter ptcmFilter=new PTCM();		
		// Non mutually exclusive false			
		ptcmFilter.setOptions(Utils.splitOptions("-M 1 -W -C -I 1 -P WORD- -Q CLUST- -L -J lexicons/AFINN-posneg.txt -H resources/50mpaths2.txt -T resources/stopwords.txt -O -B 10"));

		
		



		BufferedReader reader = new BufferedReader(
				new FileReader(inputFile));
		Instances sourceData = new Instances(reader);
		reader.close();


		System.out.println("Emoticon Results");
		wlf.processDistFilt(sourceData, emoFilter);

		System.out.println("\n\n ASA Results");
		wlf.processDistFilt(sourceData, asaFilter);
		

		System.out.println("\n\n PTCM Results");
		wlf.processDistFilt(sourceData, ptcmFilter);





	}





}
