package basic.preprocess;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;


public class GroundAllRules {
	public HashMap<String, Integer> MapRelation2ID = null;


	public HashMap<Integer, ArrayList<Integer>> LstRuleTypeI = null;
	public HashMap<Integer, ArrayList<String>> LstTrainingTriples = null;
	public ArrayList<String> TrainingTriples_list = null;
	public HashMap<String, Integer> MapVariable = null;
	public HashMap<Integer, ArrayList<String>> Relation2Tuple = null;
	public HashMap<Integer, HashMap<Integer,HashMap<Integer,Boolean>>> RelSub2Obj = null;
	public HashMap<String, Boolean> TrainingTriples= null;
	
	public GroundAllRules() {
	}
	
	public void PropositionalizeRule(
			String fnRelationIDMap, 
			String fnRuleType, 
			String fnTrainingTriples, 
			String fnOutput) throws Exception {
		readData(fnRelationIDMap,
				fnTrainingTriples);
		groundRule(fnRuleType, fnOutput);
	}
	
	public void readData(String fnRelationIDMap,
			String fnTrainingTriples) throws Exception {
		MapRelation2ID = new HashMap<String, Integer>();
		
		BufferedReader reader = new BufferedReader(new InputStreamReader(
				new FileInputStream(fnRelationIDMap), "UTF-8"));
		String line = "";
		while ((line = reader.readLine()) != null) {
			String[] tokens = line.split("\t");
			Integer iRelationID = Integer.parseInt(tokens[0]);
			String strRelation = tokens[1];
			MapRelation2ID.put(strRelation, iRelationID);
		}
		reader.close();
		
		System.out.println("Start to load soft rules......");
		Relation2Tuple = new HashMap<Integer, ArrayList<String>>();
		TrainingTriples = new HashMap<String,Boolean>();
		reader = new BufferedReader(new InputStreamReader(
				new FileInputStream(fnTrainingTriples), "UTF-8"));
		while((line = reader.readLine()) != null) {
			String[] tokens = line.split("\t");
			Integer iRelationID = Integer.parseInt(tokens[1]);
			String strValue = tokens[0] + "_" + tokens[2];
			TrainingTriples.put(line,true);		
			if (!Relation2Tuple.containsKey(iRelationID)) {
				ArrayList<String> tmpLst = new ArrayList<String>();
				tmpLst.add(strValue);
				Relation2Tuple.put(iRelationID, tmpLst);
			} else {
				Relation2Tuple.get(iRelationID).add(strValue);
			}
		}
		reader.close();
		
		RelSub2Obj = new HashMap<Integer, HashMap<Integer,HashMap<Integer,Boolean>>>();
		reader = new BufferedReader(new InputStreamReader(
				new FileInputStream(fnTrainingTriples), "UTF-8"));
		while((line = reader.readLine()) != null) {
			String[] tokens = line.split("\t");
			Integer iRelationID = Integer.parseInt(tokens[1]);
			Integer iSubjectID = Integer.parseInt(tokens[0]);
			Integer iObjectID = Integer.parseInt(tokens[2]);
			if (!RelSub2Obj.containsKey(iRelationID)) {
				HashMap<Integer,HashMap<Integer,Boolean>> tmpMap = new HashMap<Integer,HashMap<Integer,Boolean>>();
				if(!tmpMap.containsKey(iSubjectID)){
					HashMap<Integer,Boolean> tmpMap_in = new HashMap<Integer,Boolean>();
					tmpMap_in.put(iObjectID,true);
					tmpMap.put(iSubjectID, tmpMap_in);
				}
				else{
					tmpMap.get(iSubjectID).put(iObjectID,true);
					}
				RelSub2Obj.put(iRelationID, tmpMap);
			} else {
				HashMap<Integer,HashMap<Integer,Boolean>> tmpMap = RelSub2Obj.get(iRelationID);
				if(!tmpMap.containsKey(iSubjectID)){
					HashMap<Integer,Boolean> tmpMap_in = new HashMap<Integer,Boolean>();
					tmpMap_in.put(iObjectID,true);
					tmpMap.put(iSubjectID, tmpMap_in);
				}
				else{
					tmpMap.get(iSubjectID).put(iObjectID,true);
				}
			}
		}
		reader.close();
		System.out.println("Success!");
	}
	
	
	public void groundRule(
			String fnRuleType,
			String fnOutput) throws Exception {
		System.out.println("Start to propositionalize soft rules......");
		BufferedReader reader = new BufferedReader(new InputStreamReader(
				new FileInputStream(fnRuleType), "UTF-8"));
		BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(
				new FileOutputStream(fnOutput), "UTF-8"));
		
		String line = "";
		
		HashMap<String, Boolean> tmpLst = new HashMap<String, Boolean>();	
		if (!fnRuleType.equals("")) {

			while ((line = reader.readLine()) != null) {
				
				if (!line.startsWith("?"))
					continue;

				String[] bodys = line.split("=>")[0].trim().split("  ");
				String[] heads = line.split("=>")[1].trim().split("  ");
				 
				if (bodys.length == 3){
					MapVariable = new HashMap<String, Integer>();
                    String bEntity1 = bodys[0];
                    int iFstRelation = MapRelation2ID.get(bodys[1]);
                    String bEntity2 = bodys[2];		

                    String hEntity1 = heads[0];
                    int iSndRelation = MapRelation2ID.get(heads[1]);
                    String hEntity2 = heads[2].split("\t")[0];	
                    String confidence = heads[2].split("\t")[1];
                    double confi = Double.parseDouble(confidence);

                	int iSize = Relation2Tuple.get(iFstRelation).size();
        			for (int iIndex = 0; iIndex < iSize; iIndex++) {
        				String strValue = Relation2Tuple.get(iFstRelation).get(iIndex);
        				int iSubjectID = Integer.parseInt(strValue.split("_")[0]);
        				int iObjectID = Integer.parseInt(strValue.split("_")[1]);
        				MapVariable.put(bEntity1, iSubjectID);
        				MapVariable.put(bEntity2, iObjectID);
    					String strKey = "(" + iSubjectID + "\t" + iFstRelation + "\t" + iObjectID + ")\t"
    							+ "(" + MapVariable.get(hEntity1) + "\t" + iSndRelation + "\t" + MapVariable.get(hEntity2) + ")";
    					String strCons = MapVariable.get(hEntity1) + "\t" + iSndRelation + "\t" + MapVariable.get(hEntity2);
    					if(!tmpLst.containsKey(strKey) && !TrainingTriples.containsKey(strCons)) {
    						writer.write("2\t" +strKey + "\t" + confi + "\n");
    						tmpLst.put(strKey, true);
    					}
    					writer.flush();
    					MapVariable.clear();
        			}
                    

				}
				
				
				if (bodys.length == 6){
					
					MapVariable = new HashMap<String, Integer>();
                    String bEntity1 = bodys[0].trim();
                    int iFstRelation =  MapRelation2ID.get(bodys[1].trim());
                    String bEntity2 = bodys[2].trim();		

                    String bEntity3 = bodys[3].trim();
                    int iSndRelation =  MapRelation2ID.get(bodys[4].trim());
                    String bEntity4 = bodys[5].trim();	
                    
                    String hEntity1 = heads[0].trim();
                    int iTrdRelation =  MapRelation2ID.get(heads[1].trim());
                    String hEntity2 = heads[2].split("\t")[0].trim();	
                    String confidence = heads[2].split("\t")[1].trim();
                    double confi = Double.parseDouble(confidence);
                 
        			HashMap<Integer,HashMap<Integer,Boolean>> mapFstRel = RelSub2Obj.get(iFstRelation);
        			HashMap<Integer,HashMap<Integer,Boolean>> mapSndRel = RelSub2Obj.get(iSndRelation);
            		Iterator<Integer> lstEntity1 = mapFstRel.keySet().iterator();
        			while (lstEntity1.hasNext()) {
    					int iEntity1ID = lstEntity1.next();
    					MapVariable.put(bEntity1, iEntity1ID);
    					ArrayList<Integer> lstEntity2 = new ArrayList<Integer>(mapFstRel.get(iEntity1ID).keySet());
    					int iFstSize = lstEntity2.size();
    					for (int iFstIndex = 0; iFstIndex < iFstSize; iFstIndex++) {
    						int iEntity2ID = lstEntity2.get(iFstIndex);
    						MapVariable.put(bEntity1, iEntity1ID); 
    						MapVariable.put(bEntity2, iEntity2ID);
    						ArrayList<Integer> lstEntity3 = new ArrayList<Integer>(); 
    						if (MapVariable.containsKey(bEntity3) && mapSndRel.containsKey(MapVariable.get(bEntity3))){
    							lstEntity3.add(MapVariable.get(bEntity3));
    						}
    						else if(!MapVariable.containsKey(bEntity3)){
    							lstEntity3 = new ArrayList<Integer>(mapSndRel.keySet());
    						}
    						int iSndSize = lstEntity3.size();
    						for (int iSndIndex = 0; iSndIndex < iSndSize; iSndIndex++) {
    							int iEntity3ID = lstEntity3.get(iSndIndex);
    							MapVariable.put(bEntity1, iEntity1ID); 
    							MapVariable.put(bEntity2, iEntity2ID); 
    							MapVariable.put(bEntity3, iEntity3ID);       							
        						ArrayList<Integer> lstEntity4 = new ArrayList<Integer>(); 
        						if (MapVariable.containsKey(bEntity4) && mapSndRel.get(iEntity3ID).containsKey(MapVariable.get(bEntity4))){
        							lstEntity4.add(MapVariable.get(bEntity4));
        						}
        						else if(!MapVariable.containsKey(bEntity4)){
        							lstEntity4 = new ArrayList<Integer>(mapSndRel.get(iEntity3ID).keySet());
        						}
        						int iTrdSize = lstEntity4.size();
        						for (int iTrdIndex = 0; iTrdIndex < iTrdSize; iTrdIndex++) {
        							int iEntity4ID = lstEntity4.get(iTrdIndex);
        							MapVariable.put(bEntity4, iEntity4ID);
        							String infer= MapVariable.get(hEntity1) + "\t" + iTrdRelation + "\t" + MapVariable.get(hEntity2);
									String strKey = "(" + iEntity1ID + "\t" + iFstRelation + "\t" + iEntity2ID + ")\t"
											+"(" + iEntity3ID + "\t" + iSndRelation + "\t" + iEntity4ID + ")\t"
											+ "(" + MapVariable.get(hEntity1) + "\t" + iTrdRelation + "\t" + MapVariable.get(hEntity2) + ")";
									if(!tmpLst.containsKey(strKey)&&!TrainingTriples.containsKey(infer)) {
										writer.write("3\t" +strKey + "\t" + confi + "\n");
										tmpLst.put(strKey, true);
									}
        						}
        						MapVariable.clear();
    						}   
    						MapVariable.clear();
    					}
    					writer.flush();
    					MapVariable.clear();
        			}
				}	

			}
		reader.close();
		writer.close();
		}
		System.out.println("Success!");
	}
	
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		String fnRelationIDMap = "datasets\\fb15k\\relationid.txt";
		String fnRuleType = "datasets\\fb15k\\fb15k_rule";
		String fnTrainingTriples = "datasets\\fb15k\\train.txt";
        String fnOutput = "datasets\\fb15k\\groundings.txt";
		
        long startTime = System.currentTimeMillis();
        GroundAllRules generator = new GroundAllRules();
        generator.PropositionalizeRule(fnRelationIDMap, fnRuleType, fnTrainingTriples, fnOutput);
        long endTime = System.currentTimeMillis();
		System.out.println("All running time:" + (endTime-startTime)+"ms");
	}

}