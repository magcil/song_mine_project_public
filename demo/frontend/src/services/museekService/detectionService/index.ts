import { PaginatedResultDTO, PopularWinnerDTO, TopWinnersByDeviceDTO } from "./dtos/detectionModels";
import httpService from "services/httpService";
class DetectedService {
 
  public async getDetected(device?: string, page?: number, pageSize?: number): Promise<PaginatedResultDTO> {
    const params = { device, page, pageSize };
    const result = await httpService.get("/api/detected/getDetected", { params });
    return result.data;
  }

  public async getMostPopularWinner(): Promise<PopularWinnerDTO> {
    const result = await httpService.get("/api/detected/getMostPopularWinner");
    return result.data;
  }

  public async getTopWinnersByDevice(topN?: number): Promise<TopWinnersByDeviceDTO[]> {
    const params = { topN };
    const result = await httpService.get("/api/detected/getTopWinnersByDevice", { params });
    return result.data;
  }
}

export default new DetectedService();