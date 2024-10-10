import httpService from "services/httpService";

class SongsService {
  public async getTotalTracks(): Promise<any> {
    const result = await httpService.get("/api/songs/totalTracks");
    return result.data;
  }
  public async getAllTracks(skip:number = 0, limit:number=15): Promise<any> {
    const result = await httpService.get("/api/songs/getAllTracks", {params: {skip, limit}});
    return result.data;
  }
}

export default new SongsService();
