export const columns = [
  {
    title: "Name",
    dataIndex: "name",
    key: "name",
    render: (text: any, row: any) => {
      return <div>{row}</div>;
    },
  },
];
