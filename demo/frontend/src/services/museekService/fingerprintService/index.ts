import httpService from "services/httpService";


class FingerprintService {
    public async getTotalFingerprints(): Promise<any> {
        const result = await  httpService.get('/api/fingerprints/totalFingerprints');
        return result.data;
    }

}


export default new FingerprintService();