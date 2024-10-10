import httpService from "services/httpService";

class AuthService {
  public async login(username: string, password: string): Promise<any> {
    const result = await httpService.post("/api/auth/token", {
      username,
      password,
      grant_type: null,
      client_id: null,
      client_secret: null,
      scope: null,
    });

    localStorage.setItem("token", result.data.access_token);
    return result.data.access_token;
  }
  public async loggedin(): Promise<any> {
    const result = await httpService.get("/api/auth/authorization");

    return result.data.access_token;
  }
}

export default new AuthService();
