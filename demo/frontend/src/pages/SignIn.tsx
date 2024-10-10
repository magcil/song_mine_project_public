import {
  Button,
  Col,
  Form,
  Input,
  Layout,
  Row,
  Typography,
  message,
} from "antd";
import signinbg from "assets/images/img-signin.jpg";
import { useHistory } from "react-router-dom";
import authService from "services/authService";

const { Title } = Typography;
const { Footer, Content } = Layout;

const Login: React.FC = () => {
  const [messageApi, contextHolder] = message.useMessage();
  const history = useHistory();
  const warning = () => {
    messageApi.open({
      type: "error",
      content: "Failed to Login, please check your credentials",
      duration: 5,
    });
  };
  const onFinish = (values: any) => {
    authService
      .login(values.email, values.password)
      .then((res) => {
        history.push("/dashboard");
      })
      .catch((err) => {
        console.log(err);
        warning();
      });
  };

  const onFinishFailed = (errorInfo: any) => {
    console.log("Failed:", errorInfo);
  };

  return (
    <Layout className="layout-default layout-signin">
      {contextHolder}
      <Content className="signin">
        <Row gutter={[24, 0]} justify="space-around">
          <Col
            xs={{ span: 24, offset: 0 }}
            lg={{ span: 6, offset: 2 }}
            md={{ span: 12 }}
          >
            <Title className="mb-15">Sign In</Title>
            <Title className="font-regular text-muted" level={5}>
              Enter your email and password to sign in
            </Title>
            <Form
              onFinish={onFinish}
              onFinishFailed={onFinishFailed}
              layout="vertical"
              className="row-col"
            >
              <Form.Item
                className="username"
                label="Email"
                name="email"
                rules={[
                  {
                    required: true,
                    message: "Please input your email!",
                  },
                ]}
              >
                <Input placeholder="Email" />
              </Form.Item>

              <Form.Item
                className="username"
                label="Password"
                name="password"
                rules={[
                  {
                    required: true,
                    message: "Please input your password!",
                  },
                ]}
              >
                <Input.Password placeholder="Password" />
              </Form.Item>
              <Form.Item>
                <Button
                  type="primary"
                  htmlType="submit"
                  style={{ width: "100%" }}
                >
                  SIGN IN
                </Button>
              </Form.Item>
            </Form>
          </Col>
          <Col
            className="sign-img"
            style={{ padding: 12 }}
            xs={{ span: 24 }}
            lg={{ span: 12 }}
            md={{ span: 12 }}
          >
            <img src={signinbg} alt="" />
          </Col>
        </Row>
      </Content>
      <Footer>
        <div className="copyright">
          Museek Dashboard ©{new Date().getFullYear()} Created by{" "}
          <a
            href="https://labs-repos.iit.demokritos.gr/MagCIL/"
            target="_blank"
            rel="noreferrer"
          >
            MagCIL
          </a>
        </div>
      </Footer>
    </Layout>
  );
};
export default Login;
