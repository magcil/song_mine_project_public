import http from "services/httpService";
import { Result } from "./dto/result";

class PredictionsService {
  public  uploadAudio= async (input: any):Promise<Result> =>{
    const result= await  http.post("/api/transcription/blob", input);
    return result.data;
  }
}

export default new PredictionsService();
