# AWS Resource Cleanup Checklist (Strict Zero Cost Post-Demo)

Follow this checklist immediately after your college demo concludes to ensure no ongoing charges accrue in your AWS account.

---

## 1. Stop and Terminate the EC2 Instance

1. Open the [AWS Management Console](https://console.aws.amazon.com/).
2. Verify you are in the **Asia Pacific (Mumbai) `ap-south-1`** region.
3. Go to **EC2** $ightarrow$ **Instances**.
4. Select `fakenews-detector-demo`.
5. Click **Instance state** $ightarrow$ **Terminate instance**.
6. Confirm termination when prompted.
   - *Status will change from `Shutting-down` to `Terminated` within a few minutes.*

> **Important:** Terminating the instance automatically deletes the root EBS volume attached to it (by default, `DeleteOnTermination: true`).

---

## 2. Verify EBS Volume Deletion

1. In the EC2 console left navigation bar, click **Elastic Block Store** $ightarrow$ **Volumes**.
2. Look at the **State** column:
   - There should be **0 volumes** in `in-use` or `available` associated with the terminated instance.
3. If any volume remains in `available` state:
   - Select the volume $ightarrow$ Click **Actions** $ightarrow$ **Delete volume**.

---

## 3. Verify EBS Snapshot Removal

1. In the left navigation, click **Elastic Block Store** $ightarrow$ **Snapshots**.
2. Confirm there are **no manual snapshots** created from this demo.
3. If any snapshot exists:
   - Select it $ightarrow$ Click **Actions** $ightarrow$ **Delete snapshot**.

---

## 4. Release Elastic IPs (If Any Were Created)

*(If you followed the guide, no Elastic IP was created. Verify nonetheless):*

1. In the left navigation, click **Network & Security** $ightarrow$ **Elastic IPs**.
2. If any Elastic IP is listed:
   - Select it $ightarrow$ Click **Actions** $ightarrow$ **Release Elastic IP address**.
   - *AWS charges for unallocated or unattached Elastic IPs.*

---

## 5. Clean Up Key Pairs & Security Groups (Optional)

1. **Security Groups**:
   - Go to **Network & Security** $ightarrow$ **Security Groups**.
   - You may delete the demo security group (`Actions` $ightarrow$ `Delete security group`).
   - *(Security groups do not incur any charges).*
2. **Key Pairs**:
   - Go to **Network & Security** $ightarrow$ **Key Pairs**.
   - Delete `fakenews-key` if no longer needed.
   - *(Key pairs do not incur any charges).*

---

## 6. Verify Absence of Other Potentially Billable Services

Check each of these services in the Mumbai region:
- **Load Balancers** (EC2 $ightarrow$ Load Balancers): Must be **0**.
- **NAT Gateways** (VPC $ightarrow$ NAT Gateways): Must be **0**.
- **RDS Databases** (RDS $ightarrow$ Databases): Must be **0**.
- **CloudWatch Alarms** (CloudWatch $ightarrow$ Alarms): Remove any paid alarms.

---

## 7. Check AWS Billing & Cost Explorer

1. Navigate to **Billing and Cost Management** in the AWS Console.
2. Click **Bills** $ightarrow$ Review current month charges.
3. Click **Free Tier** in the left menu:
   - Check the percentage usage of:
     - `Amazon Elastic Compute Cloud - Compute hours` (750 hours/month).
     - `Amazon Elastic Compute Cloud - EBS gp3 storage` (30 GB/month).
4. Verify your forecasted charges are **$0.00**.
