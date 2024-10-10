// UserStore.js
import { makeAutoObservable } from "mobx";

class UserStore {
  users: any = [];

  constructor() {
    makeAutoObservable(this);
  }

  addUser(user: any) {
    this.users.push(user);
  }

  removeUser(user: any) {
    this.users = this.users.filter((u: any) => u !== user);
  }

  get userCount() {
    return this.users.length;
  }
}

export default UserStore;
