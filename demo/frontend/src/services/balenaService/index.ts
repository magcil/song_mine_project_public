import httpService from '../balenaHttpService'; // path to the file
import { BalenaDevices } from './dtos/balenaDevices';
import { Releases } from './dtos/balenaReleases';



class BalenaService {
    public async getBalenaDevices(): Promise<BalenaDevices[]> {
        const result = await  httpService.get('/v6/device?$filter=belongs_to__application%20eq%202037308');
        return result.data.d;
    }

    public async getAllReleases(): Promise<Releases[]> {
        const result = await httpService.get(`/v6/release?$filter=(belongs_to__application%20eq%202037308)%20and%20(status%20in%20(%27success%27,%20%27running%27))&$orderby=id%20desc&$select=id,status,commit,raw_version`);
        return result.data.d;
    }
}


export default new BalenaService();